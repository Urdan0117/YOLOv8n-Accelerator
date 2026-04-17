from heapq import nsmallest
from itertools import product

from analytical_model.eyeriss import (
    EyerissAnalyzer,
    AnalysisResult,
    EyerissHardwareParam,
    EyerissMappingParam,
    PSUM_DATA_SIZE,
)
from layer_info import Conv2DShapeParam, MaxPool2DShapeParam


class EyerissMapper:
    cnt = 0

    def __init__(
        self,
        name: str | None,
    ) -> None:
        self.name = name if name is not None else f"mapping_{EyerissMapper.cnt}"
        self.analyzer = EyerissAnalyzer(name=self.name)
        EyerissMapper.cnt += 1

    def run(
        self,
        conv2d: Conv2DShapeParam,
        maxpool: MaxPool2DShapeParam | None = None,
        num_solutions: int = 1,
        mode: str | None = None,
    ) -> list[AnalysisResult]:
        self.analyzer.conv_shape = conv2d
        self.analyzer.maxpool_shape = maxpool
        self.analyzer.mode = self.mode = mode
        results = []

        for hardware in self.generate_hardware():
            self.hardware = hardware

            for mapping in self.generate_mappings():
                self.analyzer.mapping = mapping
                res = self.analyzer.summary
                
                # 👇 絕招：直接把切塊參數綁定在這次的結果字典上
                res['m'] = mapping.m
                res['n'] = mapping.n
                res['e'] = mapping.e
                res['p'] = mapping.p
                res['q'] = mapping.q
                res['r'] = mapping.r
                res['t'] = mapping.t
                
                results.append(res)

        # 讓演算法幫我們挑出 EDP 最低（最好）的解
        if num_solutions > 0:
            results = nsmallest(num_solutions, results, key=self.evaluate)
            
        # 👇 挑出來之後，我們大聲把它印在終端機上！
        if len(results) > 0:
            best = results[0]
            layer_name = best.get("layer_name", "Layer")
            print(f"👉 [{layer_name} 最佳切塊] m:{best['m']}, n:{best['n']}, e:{best['e']}, p:{best['p']}, q:{best['q']}, r:{best['r']}, t:{best['t']}")

        return results

    """
     def run(
        self,
        conv2d: Conv2DShapeParam,
        maxpool: MaxPool2DShapeParam | None = None,
        num_solutions: int = 1,
        mode: str | None = None,
    ) -> list[AnalysisResult]:
        self.analyzer.conv_shape = conv2d
        self.analyzer.maxpool_shape = maxpool
        self.analyzer.mode = self.mode = mode
        results = []

        for hardware in self.generate_hardware():
            self.hardware = hardware

            for mapping in self.generate_mappings():
                self.analyzer.mapping = mapping
                res = self.analyzer.summary
                results.append(res)

        if num_solutions > 0:
            results = nsmallest(num_solutions, results, key=self.evaluate)
        return results
    """
   

    def evaluate(self, metrics: AnalysisResult) -> float:
        score = 0
        #! <<<========= Implement here =========>>>

        # 取得剛才 analytical model 算出來的總能量與總延遲
        energy = metrics.get("energy_total", float('inf'))
        latency = metrics.get("latency", float('inf'))
        
        # 使用 EDP (Energy-Delay Product) 作為評分標準：越低越好
        # 這樣可以同時懲罰「超耗電但很快」與「超省電但龜速」的極端設計
        score = energy * latency
        
        return score

    @property
    def hardware(self) -> EyerissHardwareParam:
        return self.analyzer.hardware

    @hardware.setter
    def hardware(self, hardware_param: EyerissHardwareParam) -> None:
        assert isinstance(hardware_param, EyerissHardwareParam)
        self.analyzer.hardware = hardware_param

    def p_avaliable(self) -> list[int]:
        p_max = self.hardware.psum_spad_size // PSUM_DATA_SIZE
        return list(range(1, p_max + 1))

    def q_avaliable(self) -> list[int]:
        q_max = self.hardware.ifmap_spad_size // self.analyzer.conv_shape.S
        return list(range(1, q_max + 1))

    def e_available(self) -> list[int]:
        hw_strips = self.hardware.pe_array_h // self.analyzer.conv_shape.R
        e_max = self.hardware.pe_array_w * hw_strips
        return list(range(1, min(e_max, self.analyzer.conv_shape.E) + 1))

    def r_available(self) -> list[int]:
        r_max = self.hardware.pe_array_h // self.analyzer.conv_shape.R
        return list(range(1, r_max + 1))

    def t_available(self) -> list[int]:
        num_pes = self.hardware.pe_array_h * self.hardware.pe_array_w
        t_max = num_pes // self.analyzer.conv_shape.R
        return list(range(1, t_max + 1))

    def m_available(self) -> list[int]:
        m_max = self.analyzer.conv_shape.M
        return list(
            m for m in range(1, m_max + 1) if self.analyzer.conv_shape.M % m == 0
        )

    def validate(self, mapping) -> bool:
        m, n, e, p, q, r, t = mapping
        self.analyzer.mapping = EyerissMappingParam(*mapping)

        # pq constraints
        if p * q > self.hardware.filter_spad_size // self.analyzer.conv_shape.S:
            return False

        # e constraints
        if (
            e % self.hardware.pe_array_w != 0
            and e != self.hardware.pe_array_w // 2
            and self.analyzer.conv_shape.E != e
        ):
            return False

        # rt constraints
        if (
            r * t
            != self.hardware.pe_array_h
            * self.hardware.pe_array_w
            // self.analyzer.conv_shape.R
            // e
        ):
            return False

        # m constraints
        if m % p != 0:
            return False

        return self.analyzer.glb_size_legal

    def generate_mappings(self, verbose: bool = False) -> list[EyerissMappingParam]:
        candidate_solutions = []
        #! <<<========= Implement here =========>>>
       
        # 1. 取得所有參數的可能範圍 (n 通常固定為 1，因為我們大多處理 batch_size=1 的 inference)
        n_available_list = [1]
        p_available_list = self.p_avaliable()
        q_available_list = self.q_avaliable()
        e_available_list = self.e_available()
        r_available_list = self.r_available()
        t_available_list = self.t_available()
        m_available_list = self.m_available()

        # 2. 利用 itertools.product 產生所有可能的排列組合 (笛卡兒乘積)
        all_combinations = product(
            m_available_list,
            n_available_list,
            e_available_list,
            p_available_list,
            q_available_list,
            r_available_list,
            t_available_list,
        )


        # 3. 逐一驗證組合是否合法 (會不會超過 SPAD/GLB 的容量，或違反架構限制)
        for sol in all_combinations:

            # 必須先把這組參數轉成 Mapping 物件，並餵給 analyzer！
            # 否則 validate() 裡面的 glb_size_legal 會抓不到數據而全部判定失敗！
            mapping = EyerissMappingParam(*sol)
            self.analyzer.mapping = mapping

            if self.validate(sol):
                # 如果合法，就轉換成 EyerissMappingParam 物件並加入候選名單
                candidate_solutions.append(EyerissMappingParam(*sol))

        return candidate_solutions

    def generate_hardware(self) -> list[EyerissHardwareParam]:
        candidate_solutions = []

        """
        # 擴展 PE Array 大小以探索 Compute-bound 解法
        pe_array_h_list = [6, 12]  
        pe_array_w_list = [8, 16]  
        
        # 擴展 Spad 與 GLB 容量以探索 Memory-bound 解法
        ifmap_spad_size_list = [12, 24]
        filter_spad_size_list = [48, 96]
        psum_spad_size_list = [16, 32]
        glb_size_list = [64 * 2**10, 128 * 2**10] 
        
        # 擴展頻寬
        bus_bw_list = [4, 8]
        noc_bw_list = [4, 8]
    
        """       
        # for AOCfinal
        pe_array_h_list = [12]  # add more values to explore more solutions
        pe_array_w_list = [16]  # add more values to explore more solutions               
        ifmap_spad_size_list = [24]
        filter_spad_size_list = [96]
        psum_spad_size_list = [32]
        glb_size_list = [128 * 2**10]
        bus_bw_list = [8]
        noc_bw_list = [8]
        
        candidate_solutions = product(
            pe_array_h_list,
            pe_array_w_list,
            ifmap_spad_size_list,
            filter_spad_size_list,
            psum_spad_size_list,
            glb_size_list,
            bus_bw_list,
            noc_bw_list,
        )
        candidate_solutions = [EyerissHardwareParam(*m) for m in candidate_solutions]
        return candidate_solutions
