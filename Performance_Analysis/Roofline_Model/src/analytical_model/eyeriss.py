from dataclasses import dataclass, asdict
from math import ceil

from layer_info import Conv2DShapeParam, MaxPool2DShapeParam

# Memory
DATA_SIZE = 1  # Byte
PSUM_DATA_SIZE = 4  # Byte
BUS_BANDWIDTH = 4  # Byte

# Time
CLOCK_RATE = 200 * 1e6  # 200 MHz
TIME_UNIT = 1  # cycle
SPAD_ACCESS_TIME = 1 * TIME_UNIT
GLB_ACCESS_TIME = 2 * TIME_UNIT
DRAM_ACCESS_TIME = 5 * TIME_UNIT

# Energy
ENERGY_UNIT = 1e-6  # 1 pJ = 10^6 uJ
ENERGY_PER_MAC = 2 * ENERGY_UNIT
ENERGY_PER_GLB_ACCESS = 10 * ENERGY_UNIT
ENERGY_PER_DRAM_ACCESS = 200 * ENERGY_UNIT
POWER_UNIT = 1  # 1 uW
POWER_LEAKAGE = 50 * POWER_UNIT

######################################################################################################
# N: number of ifmaps/ofmaps
# M: number of filters
# H/W: ifmap height/width
# R/S: filter height/width
# E/F: ofmap height/width
# U: stride
#  ----------------------------------------------------------------------------------------------
# m: ofmap channels in global buffer
# n: number of ifmaps in a pass
# e: width of PE-set
# p: number of filters in a pass
# q: (ifmap or filter) channels in a pass
# r: number of PE sets for different (ifmap/filter) channels
# t: number of PE sets for different filters
#  ----------------------------------------------------------------------------------------------
#  Naming Convention
# *_per_pass: compute / storage size required per pass
# *_per_layer: compute / storage size required per layer
######################################################################################################


@dataclass
class EyerissHardwareParam:
    pe_array_h: int
    pe_array_w: int
    ifmap_spad_size: int
    filter_spad_size: int
    psum_spad_size: int
    glb_size: int
    bus_bw: int
    noc_bw: int


@dataclass
class EyerissMappingParam:
    m: int  # number of ofmap channels stored in global buffer
    n: int  # number of ofmaps/ifmaps used in a processing pass
    e: int  # width of the PE set (strip-mined if nessary)
    p: int  # number of filters processed by a PE set
    q: int  # number of ifmap/filter channels processed by a PE set
    r: int  # number of PE sets for different ifmap/filter channels
    t: int  # number of PE sets for different filters


AnalysisResult = dict[str, str | int | float]


class EyerissAnalyzer:
    cnt = 0

    def __init__(
        self,
        name: str | None = None,
        hardware_param: EyerissHardwareParam | None = None,
    ) -> None:
        self.name = name if name is not None else f"mapping_{EyerissAnalyzer.cnt}"
        self._hardware = hardware_param
        self._conv_shape = None
        self._maxpool_shape = None
        self._mapping = None
        EyerissAnalyzer.cnt += 1

    @property
    def hardware(self) -> EyerissHardwareParam:
        return self._hardware

    @hardware.setter
    def hardware(self, hardware_param: EyerissHardwareParam) -> None:
        assert isinstance(hardware_param, EyerissHardwareParam)
        self._hardware = hardware_param

    @property
    def conv_shape(self) -> Conv2DShapeParam:
        return self._conv_shape

    @conv_shape.setter
    def conv_shape(self, conv_param: Conv2DShapeParam) -> None:
        assert isinstance(conv_param, Conv2DShapeParam)
        self._conv_shape = conv_param

    @property
    def maxpool_shape(self) -> MaxPool2DShapeParam:
        return self._maxpool_shape

    @maxpool_shape.setter
    def maxpool_shape(self, maxpool_param: MaxPool2DShapeParam | None) -> None:
        assert isinstance(maxpool_param, (MaxPool2DShapeParam, type(None)))
        self._maxpool_shape = maxpool_param

    @property
    def mapping(self) -> EyerissMappingParam:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping_param: EyerissMappingParam) -> None:
        self._mapping = mapping_param

    # Scratchpad Memory Usage
    def filter_used(self) -> int:
        return self.mapping.q * self.conv_shape.S * self.mapping.p

    def ifmap_used(self) -> int:
        return self.mapping.q * self.conv_shape.S

    def psum_used(self) -> int:
        return self.mapping.p

    @property
    def spad_size_legal(self) -> dict[str, bool]:
        return {
            "ifmap": self.ifmap_used() <= self.hardware.ifmap_spad_size,
            "filter": self.filter_used() <= self.hardware.filter_spad_size,
            "psum": self.psum_used() <= self.hardware.psum_spad_size,
        }

    @property
    def spad_usage(self) -> dict[str, int]:
        return {
            "ifmap": self.ifmap_used(),
            "filter": self.filter_used(),
            "psum": self.psum_used(),
        }

    # Global Buffer (GLB) Usage
    @property
    def glb_usage_per_pass(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        #! <<<========= Implement here =========>>>

        N, H, W = self.conv_shape.N, self.conv_shape.H, self.conv_shape.W
        R, S = self.conv_shape.R, self.conv_shape.S
        E, F = self.conv_shape.E, self.conv_shape.F
        C, M = self.conv_shape.C, self.conv_shape.M
        U = self.conv_shape.U

        m, n, e = self.mapping.m, self.mapping.n, self.mapping.e
        p, q, r, t = self.mapping.p, self.mapping.q, self.mapping.r, self.mapping.t

        # 計算 Ifmap 在一個 Pass 佔用的 GLB 空間：
        # n: Batch 數量 | (q * r): 一次處理的輸入通道數
        # (U * (e - 1) + R): 這是「感受野 (Receptive Field)」公式！考慮了 Stride (U) 後，要算出長度為 e 的輸出，需要多少高度的輸入。
        # W: 硬體特性，通常 SRAM 讀取是整行 (Row) 讀取，所以直接乘上完整寬度 W。

        # ifmap
        sizes["ifmap"] = n * (q * r) * (U * (e - 1) + R) * W * DATA_SIZE

        # filter
        # (p * t): 總共並行處理的輸出通道 (Filters) | (q * r): 總共並行的輸入通道 | R * S: 卷積核面積
        sizes["filter"] = (p * t) * (q * r) * R * S * DATA_SIZE

        # bias
        # 每個被處理的輸出通道 (p * t) 都有一個 Bias，其資料大小與 Psum 相同 (4 Bytes)
        sizes["bias"] = (p * t) * PSUM_DATA_SIZE


        # psum / ofmap
        # GLB 會保留 m 個輸出通道的空間來累積 Psum。長度為 e，並乘以完整寬度 F。
        sizes["psum"] = n * m * e * F * PSUM_DATA_SIZE

        sizes["total"] = sum(sizes.values())
        return sizes
                

    
    @property
    def glb_size_legal(self) -> bool:
        return self.glb_usage_per_pass["total"] <= self.hardware.glb_size

    # DRAM Accesses (DRAM-GLB data movement)
    @property
    def dram_access_per_layer(self) -> dict[str, int]:
        res: dict[str, int] = {}
        #! <<<========= Implement here =========>>>

        N, H, W = self.conv_shape.N, self.conv_shape.H, self.conv_shape.W
        R, S = self.conv_shape.R, self.conv_shape.S
        E, F = self.conv_shape.E, self.conv_shape.F
        C, M = self.conv_shape.C, self.conv_shape.M
        U = self.conv_shape.U

        m, n, e = self.mapping.m, self.mapping.n, self.mapping.e
        p, q, r, t = self.mapping.p, self.mapping.q, self.mapping.r, self.mapping.t

        # tile counts
        # 計算各個維度需要被切分成多少個 "Tile" (區塊) 才能跑完一整層：
        num_m = ceil(M / m)           # 總 Filter 數 / GLB 能容納的 Filter 數
        num_e = ceil(E / e)           # 總高度 / 每次處理的高度
        num_n = ceil(N / n)           # 總 Batch / 每次處理的 Batch
        num_c = ceil(C / (q * r))     # 總輸入通道 / 每次處理的輸入通道
        num_mt = ceil(M / (p * t))    # 總 Filter 數 / PE Array 一次能算的 Filter 數

        # tile sizes
        # 單個 Tile 的資料大小 (與前一個函數的邏輯相同)
        ifmap_tile = n * (q * r) * (U * (e - 1) + R) * W * DATA_SIZE
        filter_tile = (p * t) * (q * r) * R * S * DATA_SIZE
        bias_tile = (p * t) * PSUM_DATA_SIZE

        # DRAM reads
        # Ifmap: 如果 GLB 裝不下所有 Filter (num_m > 1)，Ifmap 就要從 DRAM 重複讀取！
        res["ifmap_read"] = num_m * num_e * num_n * num_c * ifmap_tile
        # Filter: 隨著空間位置移動 (num_e) 或換下一張圖 (num_n)，Filter 也要從 DRAM 重拿。
        res["filter_read"] = num_e * num_n * num_c * num_mt * filter_tile
        res["bias_read"] = num_e * num_n * num_c * num_mt * bias_tile

        # ofmap write
        ofmap = N * M * E * F
        # 如果有 Pooling 層，寫回 DRAM 的資料量會依照 stride 的平方倍縮小
        if self.maxpool_shape is not None:
            ofmap = ofmap // (self.maxpool_shape.stride ** 2)

        res["write"] = ofmap * DATA_SIZE

        res["read"] = res["ifmap_read"] + res["filter_read"] + res["bias_read"]
        res["total"] = res["read"] + res["write"]

        return res
        """
        # Ifmap 會因為 m (GLB能放的ofmap數) 不夠大，需要重複讀取
        num_m_passes = ceil(self.conv_shape.M / self.mapping.m)
        res["ifmap_read"] = self.conv_shape.N * self.conv_shape.C * self.conv_shape.H * self.conv_shape.W * num_m_passes * DATA_SIZE
        
        # Filter 會因為 e (每次處理的寬度) 不夠大，需要重複讀取
        num_n_passes = ceil(self.conv_shape.N / self.mapping.n)
        num_f_passes = ceil(self.conv_shape.F / self.mapping.e)
        res["filter_read"] = self.conv_shape.M * self.conv_shape.C * self.conv_shape.R * self.conv_shape.S * num_n_passes * num_f_passes * DATA_SIZE
        
        res["bias_read"] = self.conv_shape.M * num_n_passes * DATA_SIZE
        
        # Ofmap 寫回 (考慮 MaxPool 降維)
        ofmap_size = self.conv_shape.N * self.conv_shape.M * self.conv_shape.E * self.conv_shape.F
        if self.maxpool_shape is not None:
            ofmap_size = ofmap_size // (self.maxpool_shape.stride ** 2)
            
        res["read"] = res["ifmap_read"] + res["filter_read"] + res["bias_read"]
        res["write"] = ofmap_size * DATA_SIZE
        res["total"] = res["read"] + res["write"]
        return res
        """
        
    

    # GLB Accesses (GLB-Spad data movement)
    @property
    def glb_access_per_layer(self) -> dict[str, int]:
        res: dict[str, int] = {}
        #! <<<========= Implement here =========>>>
        N, H, W = self.conv_shape.N, self.conv_shape.H, self.conv_shape.W
        R, S = self.conv_shape.R, self.conv_shape.S
        E, F = self.conv_shape.E, self.conv_shape.F
        C, M = self.conv_shape.C, self.conv_shape.M
        U = self.conv_shape.U

        m, n, e = self.mapping.m, self.mapping.n, self.mapping.e
        p, q, r, t = self.mapping.p, self.mapping.q, self.mapping.r, self.mapping.t

        # tile counts
        # 一樣先計算各維度的切塊數量
        num_m = ceil(M / m)
        num_e = ceil(E / e)
        num_n = ceil(N / n)
        num_c = ceil(C / (q * r))
        num_mt = ceil(M / (p * t))

        # tile sizes
        ifmap_tile = n * (q * r) * (U * (e - 1) + R) * W * DATA_SIZE
        filter_tile = (p * t) * (q * r) * R * S * DATA_SIZE
        bias_tile = (p * t) * PSUM_DATA_SIZE

        # reuse
        # Filter 在 GLB 內部的重用次數 (Reuse)：
        # GLB 一次存了 m 個通道，但 PE 陣列一次只能吃 p*t 個。
        # 代表這批 Ifmap 在 GLB 裡面可以餵給 PE 陣列 reuse 次，來算不同的 Filter！
        reuse = ceil(m / (p * t))

        # GLB read
        res["ifmap_read"] = num_m * num_e * num_n * num_c * reuse * ifmap_tile
        res["filter_read"] = num_e * num_n * num_c * num_mt * filter_tile
        res["bias_read"] = num_e * num_n * num_c * num_mt * bias_tile

        # psum (conv)
        ofmap = N * M * E * F
        # 如果輸入通道 C 被切成 num_c 塊。除了第 1 塊不用讀舊資料，剩下的 (num_c - 1) 塊都要從 GLB 讀出先前的 Psum 來累加。
        res["psum_read_conv"] = (num_c - 1) * ofmap * PSUM_DATA_SIZE
        # 每算完一個 C 區塊，就要寫回 GLB 一次，總共寫 num_c 次。
        res["psum_write_conv"] = num_c * ofmap * PSUM_DATA_SIZE

        # Pooling 的額外存取：如果包含 MaxPool，需要把算好的 Psum 從 GLB 讀出來做比較，再把降維後的結果寫回去。
        if self.maxpool_shape is not None:
            pooled = ofmap // (self.maxpool_shape.stride ** 2)
            res["psum_read_pool"] = ofmap * PSUM_DATA_SIZE
            res["ofmap_write_pool"] = pooled * DATA_SIZE
        else:
            res["psum_read_pool"] = 0
            res["ofmap_write_pool"] = 0

        res["psum_read"] = res["psum_read_conv"] + res["psum_read_pool"]
        res["psum_write"] = res["psum_write_conv"] + res["ofmap_write_pool"]

        res["read"] = res["ifmap_read"] + res["filter_read"] + res["bias_read"] + res["psum_read"]
        res["write"] = res["psum_write"]
        res["total"] = res["read"] + res["write"]

        return res

        """
        num_p_passes = ceil(self.conv_shape.M / (self.mapping.p * self.mapping.t))
                num_c_passes = ceil(self.conv_shape.C / (self.mapping.q * self.mapping.r))
                num_n_passes = ceil(self.conv_shape.N / self.mapping.n)
                num_f_passes = ceil(self.conv_shape.F / self.mapping.e)
                
                # Filter 讀取：每次換寬度區塊 (F/e) 和換 Batch (N/n) 時重拿
                res["filter_read"] = self.conv_shape.M * self.conv_shape.C * self.conv_shape.R * self.conv_shape.S * num_n_passes * num_f_passes * DATA_SIZE
                
                # Ifmap 讀取：每次換一批新的 Filter (M/p*t) 時重拿
                res["ifmap_read"] = self.conv_shape.N * self.conv_shape.C * self.conv_shape.H * self.conv_shape.W * num_p_passes * DATA_SIZE
                
                res["bias_read"] = self.conv_shape.M * DATA_SIZE
                
                # Psum 累積 (如果 C 被切分，中間值要寫回 GLB 再讀出來)
                total_ofmap_size = self.conv_shape.N * self.conv_shape.M * self.conv_shape.E * self.conv_shape.F
                res["psum_read_conv"] = total_ofmap_size * (num_c_passes - 1) * PSUM_DATA_SIZE
                res["psum_write_conv"] = total_ofmap_size * num_c_passes * PSUM_DATA_SIZE
                
                # Pooling 的額外讀寫
                res["psum_read_pool"] = 0 if self.maxpool_shape is None else total_ofmap_size * PSUM_DATA_SIZE
                res["ofmap_write_pool"] = 0 if self.maxpool_shape is None else (total_ofmap_size // (self.maxpool_shape.stride ** 2)) * DATA_SIZE
                
                res["psum_read"] = res["psum_read_conv"] + res["psum_read_pool"]
                res["psum_write"] = res["psum_write_conv"] + res["ofmap_write_pool"]
                
                res["read"] = res["ifmap_read"] + res["filter_read"] + res["bias_read"] + res["psum_read"]
                res["write"] = res["psum_write"]
                res["total"] = res["read"] + res["write"]
                
                return res
        """
        
    
    @property
    def latency_per_layer(self) -> int:
        ofmap_size = (
            self.conv_shape.N
            * self.conv_shape.M
            * self.conv_shape.E
            * self.conv_shape.F
        )
        ppu_latency_per_elem = 1 if self.maxpool_shape is None else 5

        return (
            ceil(
                self.glb_access_per_layer["total"]
                * GLB_ACCESS_TIME
                / self.hardware.noc_bw
            )
            + ceil(
                self.dram_access_per_layer["total"]
                * DRAM_ACCESS_TIME
                / self.hardware.bus_bw
            )
            + ofmap_size * ppu_latency_per_elem
        )

    @property
    def macs_per_layer(self) -> int:
        return (
            self.conv_shape.N
            * self.conv_shape.M
            * self.conv_shape.E
            * self.conv_shape.F
            * self.conv_shape.C
            * self.conv_shape.R
            * self.conv_shape.S
        )

    @property
    def energy_per_layer(self) -> dict[str, float]:
        compute_energy = self.macs_per_layer * ENERGY_PER_MAC
        memory_energy = (
            self.glb_access_per_layer["total"] * ENERGY_PER_GLB_ACCESS
            + self.dram_access_per_layer["total"] * ENERGY_PER_DRAM_ACCESS
        )
        leakage_energy = POWER_LEAKAGE * self.latency_per_layer / CLOCK_RATE
        total_energy = compute_energy + memory_energy + leakage_energy
        return {
            "compute": compute_energy,
            "memory": memory_energy,
            "leakage": leakage_energy,
            "total": total_energy,
        }

    @property
    def power_per_layer(self) -> dict[str, float]:
        compute_power = (
            self.energy_per_layer["compute"] / self.latency_per_layer * CLOCK_RATE
        )
        memory_power = (
            self.energy_per_layer["memory"] / self.latency_per_layer * CLOCK_RATE
        )
        leakage_power = POWER_LEAKAGE
        total_power = compute_power + memory_power + leakage_power
        return {
            "compute": compute_power,
            "memory": memory_power,
            "leakage": leakage_power,
            "total": total_power,
        }

    @property
    def operational_intensity(self) -> float:
        return self.macs_per_layer / self.dram_access_per_layer["total"]

    @property
    def peak_performance(self) -> float:
        return self.hardware.pe_array_h * self.hardware.pe_array_w  # MACs per cycle

    @property
    def peak_bandwidth(self) -> float:
        return self.hardware.bus_bw  # bytes per cycle

    @property
    def bound_by(self) -> str:
        machine_blance_point = self.peak_performance / self.peak_bandwidth
        if self.operational_intensity > machine_blance_point:
            return "compute"
        elif self.operational_intensity < machine_blance_point:
            return "memory"
        else:
            return "balanced"

    @property
    def is_compute_bound(self) -> bool:
        return self.bound_by == "compute"

    @property
    def is_memory_bound(self) -> bool:
        return self.bound_by == "memory"

    @property
    def is_balanced(self) -> bool:
        return self.bound_by == "balanced"

    @property
    def summary(self) -> AnalysisResult:
        if not hasattr(self, "mode"):
            self.mode = None

        match self.mode:
            case _:
                return {
                    "layer": self.name,
                    "glb_usage": self.glb_usage_per_pass["total"],  # bytes

                    "glb_ifmap_read": self.glb_access_per_layer["ifmap_read"],  # bytes
                    "glb_filter_read": self.glb_access_per_layer["filter_read"],  # bytes
                    "glb_bias_read": self.glb_access_per_layer["bias_read"],  # bytes
                    "glb_psum_read_conv": self.glb_access_per_layer["psum_read_conv"],  # bytes
                    "glb_psum_read_pool": self.glb_access_per_layer["psum_read_pool"],  # bytes
                    "glb_psum_read": self.glb_access_per_layer["psum_read"],  # bytes
                    "glb_psum_write_conv": self.glb_access_per_layer["psum_write_conv"],  # bytes
                    "glb_ofmap_write_pool": self.glb_access_per_layer["ofmap_write_pool"],  # bytes
                    "glb_psum_write": self.glb_access_per_layer["psum_write"],  # bytes

                    "glb_read": self.glb_access_per_layer["read"],  # bytes
                    "glb_write": self.glb_access_per_layer["write"],  # bytes
                    "glb_access": self.glb_access_per_layer["total"],  # bytes

                    "dram_ifmap_read": self.dram_access_per_layer["ifmap_read"],  # bytes
                    "dram_filter_read": self.dram_access_per_layer["filter_read"],  # bytes
                    "dram_bias_read": self.dram_access_per_layer["bias_read"],  # bytes
                    "dram_write": self.dram_access_per_layer["read"],  # bytes
                    "dram_write": self.dram_access_per_layer["write"],  # bytes
                    "dram_access": self.dram_access_per_layer["total"],  # bytes
                    "macs": self.macs_per_layer,
                    "latency": self.latency_per_layer,  # cycles
                    "energy_total": self.energy_per_layer["total"],  # uJ
                    "power_total": self.power_per_layer["total"],  # uW
                    # or any other metrics you want to include in the report
                    # Roofline Model 需要的數據 
                    "peak_performance": self.hardware.pe_array_h * self.hardware.pe_array_w, # 總 PE 數量 = 理論最大運算量 (MACs/cycle)
                    "peak_bandwidth": self.hardware.bus_bw, # 外部匯流排頻寬 = 理論最大頻寬 (Bytes/cycle)
                    "performance": self.macs_per_layer / self.latency_per_layer if self.latency_per_layer > 0 else 0, # 實際算力 (MACs/cycle)
                    "intensity": self.macs_per_layer / self.dram_access_per_layer["total"] if self.dram_access_per_layer["total"] > 0 else 0, # 算術強度 (MACs/Byte)
                }
