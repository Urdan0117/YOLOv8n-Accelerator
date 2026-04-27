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
        conv2d,
        maxpool = None,
        num_solutions: int = 1,
        mode: str | None = None,
    ):
        self.analyzer.conv_shape = conv2d
        self.analyzer.maxpool_shape = maxpool
        self.analyzer.mode = self.mode = mode
        results = []

        for mapping in self.generate_mappings():
            self.analyzer.mapping = mapping
            
            # 🌟 終極防禦：強制深拷貝，避免字典指標污染 (Reference Bug)！
            try:
                res = self.analyzer.summary.copy()
            except AttributeError:
                res = dict(self.analyzer.summary)
                
            res['m'] = mapping.m
            res['n'] = mapping.n
            res['e'] = mapping.e
            res['p'] = mapping.p
            res['q'] = mapping.q
            res['r'] = mapping.r
            res['t'] = mapping.t
            
            results.append(res)

        from heapq import nsmallest
        if num_solutions > 0 and results:
            results = nsmallest(num_solutions, results, key=self.evaluate)
            
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

    def generate_mappings(self, verbose: bool = False):
        from analytical_model.eyeriss import EyerissMappingParam
        if not hasattr(self.analyzer.conv_shape, 'M'):
            return [EyerissMappingParam(1, 1, 1, 1, 1, 1, 1)]

        pe_h = self.hardware.pe_array_h
        pe_w = self.hardware.pe_array_w
        R = self.analyzer.conv_shape.R
        E = self.analyzer.conv_shape.E
        S = self.analyzer.conv_shape.S
        
        candidate_solutions = []

        # 1. 抓出所有可能的 e
        e_cands = set()
        for e in range(1, E + 1):
            if e % pe_w == 0 or e == pe_w // 2 or e == E:
                e_cands.add(e)
        e_cands = sorted(list(e_cands), reverse=True)

        # 🌟 核心引擎：算出硬體的物理極限，用來做超高速攔截
        p_q_max = self.hardware.filter_spad_size // S

        for e in e_cands:
            rt_product = (pe_h * pe_w) // R // e
            if rt_product <= 0: continue
            
            rt_pairs = []
            for r in self.r_available():
                if rt_product % r == 0:
                    t = rt_product // r
                    if t in self.t_available():
                        rt_pairs.append((r, t))
            if not rt_pairs: continue

            # 🌟 解除封印：不再使用 [:3]，把所有的 p 和 q 全部拿出來測！
            p_cands = sorted(self.p_avaliable(), reverse=True)
            q_cands = sorted(self.q_avaliable(), reverse=True)

            for r, t in rt_pairs:
                for p in p_cands:
                    for q in q_cands:
                        # 🚀 超高速 O(1) 攔截：如果 p*q 太大，連模擬器都不用進，直接跳過 (這拯救了 90% 的運算時間)
                        if p * q > p_q_max:
                            continue
                            
                        # 找出能被 p 整除的通道數 m
                        m_valid = sorted([m for m in self.m_available() if m % p == 0], reverse=True)
                        if not m_valid: m_valid = [1]
                            
                        for m in m_valid:
                            sol = (m, 1, e, p, q, r, t)
                            # 只有通過上方數學篩選的菁英，才真正丟給模擬器去算 GLB (最耗時的步驟)
                            self.analyzer.mapping = EyerissMappingParam(*sol)
                            if self.analyzer.glb_size_legal:
                                candidate_solutions.append(EyerissMappingParam(*sol))
                                # 🌟 找到這個 p,q 下最大且合法的 m 後，就換下一組 p,q，避免陷入泥沼
                                break  

        # 防呆保底
        if not candidate_solutions:
            return [EyerissMappingParam(1, 1, 1, 1, 1, 1, 1)]

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
        pe_array_h_list = [12, 16]  # add more values to explore more solutions
        pe_array_w_list = [16]  # add more values to explore more solutions               
        ifmap_spad_size_list = [24]
        filter_spad_size_list = [96]
        psum_spad_size_list = [32]
        glb_size_list = [128 * 2**10, 256 * 2**10]
        bus_bw_list = [8, 16]
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
