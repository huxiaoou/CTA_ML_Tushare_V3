import itertools as ittl
from typing import NewType, Literal
from enum import StrEnum
from dataclasses import dataclass
from husfort.qsqlite import CDbStruct

"""
----------------------------
Part I: test returns
----------------------------
"""

TReturnClass = NewType("TReturnClass", str)
TReturnName = NewType("TReturnName", str)
TReturnNames = list[TReturnName]


class TRetPrc(StrEnum):
    OPN = "Opn"
    CLS = "Cls"


@dataclass(frozen=True)
class CRet:
    ret_prc: TRetPrc
    win: int
    lag: int

    @property
    def shift(self) -> int:
        return self.win + self.lag

    @property
    def ret_class(self) -> TReturnClass:
        return TReturnClass(f"{self.win:03d}L{self.lag:d}")

    @property
    def ret_name(self) -> TReturnName:
        return TReturnName(f"{self.ret_prc}{self.win:03d}L{self.lag:d}")

    @property
    def save_id(self) -> str:
        return f"{self.win:03d}L{self.lag:d}"

    @staticmethod
    def parse_from_name(return_name: str) -> "CRet":
        ret_prc = TRetPrc(return_name[0:3])
        win = int(return_name[3:6])
        lag = int(return_name[7])
        return CRet(ret_prc=ret_prc, win=win, lag=lag)


TRets = list[CRet]

"""
----------------------------------
Part II: factors configuration
----------------------------------
"""

TFactorClass = NewType("TFactorClass", str)
TFactorName = NewType("TFactorName", str)
TFactorNames = list[TFactorName]


@dataclass(frozen=True)
class CFactor:
    factor_class: TFactorClass
    factor_name: TFactorName


TFactors = list[CFactor]


@dataclass(frozen=True)
class CCfgFactor:
    @property
    def factor_class(self) -> TFactorClass:
        raise NotImplementedError

    @property
    def factor_names(self) -> TFactorNames:
        raise NotImplementedError

    def get_factors(self) -> TFactors:
        res = [CFactor(self.factor_class, factor_name) for factor_name in self.factor_names]
        return TFactors(res)


# cfg for factors
@dataclass(frozen=True)
class CCfgFactorMTM(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("MTM")

    @property
    def factor_names(self) -> TFactorNames:
        return TFactorNames([TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins])


@dataclass(frozen=True)
class CCfgFactorSKEW(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SKEW")

    @property
    def factor_names(self) -> TFactorNames:
        return TFactorNames([TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins])


@dataclass(frozen=True)
class CCfgFactorKURT(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("KURT")

    @property
    def factor_names(self) -> TFactorNames:
        return TFactorNames([TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins])


@dataclass(frozen=True)
class CCfgFactorRS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("RS")

    @property
    def factor_names(self) -> TFactorNames:
        rspa = [TFactorName(f"{self.factor_class}PA{w:03d}") for w in self.wins]
        rsla = [TFactorName(f"{self.factor_class}LA{w:03d}") for w in self.wins]
        rsdif = [TFactorName(f"{self.factor_class}DIF")]
        return TFactorNames(rspa + rsla + rsdif)


@dataclass(frozen=True)
class CCfgFactorBASIS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("BASIS")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        # n1 = [TFactorName(f"{self.factor_class}D{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorTS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("TS")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}D{w:03d}") for w in self.wins]
        return TFactorNames(n0 + n1)


@dataclass(frozen=True)
class CCfgFactorS0BETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("S0BETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        # n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        n2 = [f"{self.factor_class}{w:03d}RES" for w in self.wins]
        # n3 = [f"{self.factor_class}{w:03d}RESSTD" for w in self.wins]
        return TFactorNames(n0 + n2)


@dataclass(frozen=True)
class CCfgFactorS1BETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("S1BETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        # n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        n2 = [f"{self.factor_class}{w:03d}RES" for w in self.wins]
        # n3 = [f"{self.factor_class}{w:03d}RESSTD" for w in self.wins]
        return TFactorNames(n0 + n2)


@dataclass(frozen=True)
class CCfgFactorCBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CBETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        # n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [f"{self.factor_class}{w:03d}RES" for w in self.wins]
        # n3 = [f"{self.factor_class}{w:03d}RESSTD" for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorIBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("IBETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        # n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [f"{self.factor_class}{w:03d}RES" for w in self.wins]
        # n3 = [f"{self.factor_class}{w:03d}RESSTD" for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorPBETA(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("PBETA")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        # n1 = [TFactorName(f"{self.factor_class}{self.wins[0]:03d}D{w:03d}") for w in self.wins[1:]]
        # n2 = [f"{self.factor_class}{w:03d}RES" for w in self.wins]
        # n3 = [f"{self.factor_class}{w:03d}RESSTD" for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorCTP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CTP")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t * 10):02d}") for w, t in
              ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorCTR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CTR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t * 10):02d}") for w, t in
              ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorCVP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CVP")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t * 10):02d}") for w, t in
              ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorCVR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CVR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t * 10):02d}") for w, t in
              ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorCSP(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CSP")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t * 10):02d}") for w, t in
              ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorCSR(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("CSR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t * 10):02d}") for w, t in
              ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorCOV(CCfgFactor):
    wins: list[int]
    tops: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("COV")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{int(t * 10):02d}") for w, t in
              ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorNOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("NOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        n1 = [TFactorName(f"{self.factor_class}DIF")]
        return TFactorNames(n0 + n1)


@dataclass(frozen=True)
class CCfgFactorNDOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("NDOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorWNOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("WNOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorWNDOI(CCfgFactor):
    wins: list[int]
    tops: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("WNDOI")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}T{t:02d}") for w, t in ittl.product(self.wins, self.tops)]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorSPDWEB(CCfgFactor):
    props: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SPDWEB")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{int(prop * 100):02d}") for prop in self.props]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorSIZE(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SIZE")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorHR(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("HR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorSR(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorLIQUIDITY(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("LIQUIDITY")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorVSTD(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("VSTD")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorAMP(CCfgFactor):
    wins: list[int]
    lbds: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("AMP")

    @property
    def factor_names(self) -> TFactorNames:
        # nh = [TFactorName(f"{self.factor_class}{w:03d}T{int(l * 10):02d}H") for w, l in
        #       ittl.product(self.wins, self.lbds)]
        nl = [TFactorName(f"{self.factor_class}{w:03d}T{int(l * 10):02d}L") for w, l in
              ittl.product(self.wins, self.lbds)]
        nd = [TFactorName(f"{self.factor_class}{w:03d}T{int(l * 10):02d}D") for w, l in
              ittl.product(self.wins, self.lbds)]
        return TFactorNames(nl + nd)


@dataclass(frozen=True)
class CCfgFactorEXR(CCfgFactor):
    wins: list[int]
    dfts: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("EXR")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"DXR{w:03d}D{d:02d}") for w, d in ittl.product(self.wins, self.dfts)]
        n2 = [TFactorName(f"AXR{w:03d}D{d:02d}") for w, d in ittl.product(self.wins, self.dfts)]
        return TFactorNames(n0 + n1 + n2)


@dataclass(frozen=True)
class CCfgFactorSMT(CCfgFactor):
    lbds: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("SMT")

    @property
    def factor_names(self) -> TFactorNames:
        n_prc = [TFactorName(f"{self.factor_class}T{int(lbd * 10):02d}P") for lbd in self.lbds]
        n_ret = [TFactorName(f"{self.factor_class}T{int(lbd * 10):02d}R") for lbd in self.lbds]
        return TFactorNames(n_prc + n_ret)


@dataclass(frozen=True)
class CCfgFactorRWTC(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("RWTC")

    @property
    def factor_names(self) -> TFactorNames:
        nu = [TFactorName(f"{self.factor_class}{w:03d}U") for w in self.wins]
        nd = [TFactorName(f"{self.factor_class}{w:03d}D") for w in self.wins]
        # nt = [TFactorName(f"{self.factor_class}{w:03d}T") for w in self.wins]
        nv = [TFactorName(f"{self.factor_class}{w:03d}V") for w in self.wins]
        return TFactorNames(nu + nd + nv)


@dataclass(frozen=True)
class CCfgFactorTAILS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("TAILS")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins] + [TFactorName(f"{self.factor_class}DIF")]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorHEADS(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("HEADS")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorTOPS(CCfgFactor):
    ratios: list[float]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("TOPS")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{int(ratio * 100):02d}") for ratio in self.ratios]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorDOV(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("DOV")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}MEAN{w:02d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}STD{w:02d}") for w in self.wins]
        n2 = [TFactorName(f"{self.factor_class}RATIO{w:02d}") for w in self.wins]
        return TFactorNames(n0 + n1 + n2)


@dataclass(frozen=True)
class CCfgFactorRES(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("RES")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        n1 = [TFactorName(f"{self.factor_class}DIF")]
        return TFactorNames(n0 + n1)


@dataclass(frozen=True)
class CCfgFactorVOL(CCfgFactor):
    wins: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("VOL")

    @property
    def factor_names(self) -> TFactorNames:
        n0 = [TFactorName(f"{self.factor_class}{w:03d}") for w in self.wins]
        return TFactorNames(n0)


@dataclass(frozen=True)
class CCfgFactorTA(CCfgFactor):
    macd: tuple[int, int, int]
    bbands: tuple[int, int, int]
    sar: tuple[float, float]
    adx: int
    bop: None
    cci: int
    cmo: int
    rsi: int
    mfi: int
    willr: int
    adosc: tuple[int, int]
    obv: int
    natr: int
    ma3: list[int]

    @property
    def factor_class(self) -> TFactorClass:
        return TFactorClass("TA")

    @property
    def name_macd(self) -> TFactorName:
        fast, slow, diff = self.macd
        return TFactorName(f"{self.factor_class}MACDF{fast}S{slow}D{diff}")

    @property
    def name_bbands(self) -> TFactorName:
        timeperiod, up, dn = self.bbands
        return TFactorName(f"{self.factor_class}BBANDT{timeperiod}U{up}D{dn}")

    @property
    def name_sar(self) -> TFactorName:
        acceleration, maximum = self.sar
        return TFactorName(f"{self.factor_class}SARA{int(acceleration * 100):02d}M{int(maximum * 100):02d}")

    @property
    def name_adx(self) -> TFactorName:
        timeperiod = self.adx
        return TFactorName(f"{self.factor_class}ADXT{timeperiod}")

    @property
    def name_bop(self) -> TFactorName:
        return TFactorName(f"{self.factor_class}BOP")

    @property
    def name_cci(self) -> TFactorName:
        timeperiod = self.cci
        return TFactorName(f"{self.factor_class}CCIT{timeperiod}")

    @property
    def name_cmo(self) -> TFactorName:
        timeperiod = self.cmo
        return TFactorName(f"{self.factor_class}CMOT{timeperiod}")

    @property
    def name_rsi(self) -> TFactorName:
        timeperiod = self.rsi
        return TFactorName(f"{self.factor_class}RSIT{timeperiod}")

    @property
    def name_mfi(self) -> TFactorName:
        timeperiod = self.mfi
        return TFactorName(f"{self.factor_class}MFIT{timeperiod}")

    @property
    def name_willr(self) -> TFactorName:
        timeperiod = self.willr
        return TFactorName(f"{self.factor_class}WILLRT{timeperiod}")

    @property
    def name_adosc(self) -> TFactorName:
        fast, slow = self.adosc
        return TFactorName(f"{self.factor_class}ADOSCF{fast}S{slow}")

    @property
    def name_obv(self) -> TFactorName:
        timeperiod = self.obv
        return TFactorName(f"{self.factor_class}OBVT{timeperiod}")

    @property
    def name_natr(self) -> TFactorName:
        timeperiod = self.natr
        return TFactorName(f"{self.factor_class}NATRT{timeperiod}")

    @property
    def name_ma3(self) -> TFactorName:
        fast, mid, slow = self.ma3
        return TFactorName(f"{self.factor_class}F{fast}M{mid}S{slow}")

    @property
    def factor_names(self) -> TFactorNames:
        names_ta = [
            self.name_macd, self.name_bbands, self.name_sar, self.name_adx,
            self.name_bop, self.name_cci, self.name_cmo, self.name_rsi, self.name_mfi,
            self.name_willr, self.name_adosc, self.name_obv, self.name_natr,
            self.name_ma3,
        ]
        return TFactorNames(names_ta)


@dataclass(frozen=True)
class CCfgFactors:
    MTM: CCfgFactorMTM | None
    SKEW: CCfgFactorSKEW | None
    KURT: CCfgFactorKURT | None
    RS: CCfgFactorRS | None
    BASIS: CCfgFactorBASIS | None
    TS: CCfgFactorTS | None
    S0BETA: CCfgFactorS0BETA | None
    S1BETA: CCfgFactorS1BETA | None
    CBETA: CCfgFactorCBETA | None
    IBETA: CCfgFactorIBETA | None
    PBETA: CCfgFactorPBETA | None
    CTP: CCfgFactorCTP | None
    CTR: CCfgFactorCTR | None
    CVP: CCfgFactorCVP | None
    CVR: CCfgFactorCVR | None
    CSP: CCfgFactorCSP | None
    CSR: CCfgFactorCSR | None
    COV: CCfgFactorCOV | None
    NOI: CCfgFactorNOI | None
    NDOI: CCfgFactorNDOI | None
    WNOI: CCfgFactorWNOI | None
    WNDOI: CCfgFactorWNDOI | None
    SPDWEB: CCfgFactorSPDWEB | None
    SIZE: CCfgFactorSIZE | None
    HR: CCfgFactorHR | None
    SR: CCfgFactorSR | None
    LIQUIDITY: CCfgFactorLIQUIDITY | None
    VSTD: CCfgFactorVSTD | None
    AMP: CCfgFactorAMP | None
    EXR: CCfgFactorEXR | None
    SMT: CCfgFactorSMT | None
    RWTC: CCfgFactorRWTC | None
    TAILS: CCfgFactorTAILS | None
    HEADS: CCfgFactorHEADS | None
    TOPS: CCfgFactorTOPS | None
    DOV: CCfgFactorDOV | None
    RES: CCfgFactorRES | None
    VOL: CCfgFactorVOL | None
    TA: CCfgFactorTA | None

    def values(self) -> list[CCfgFactor]:
        res = []
        for _, v in vars(self).items():
            if v is not None:
                res.append(v)
        return res

    def get_factors(self) -> TFactors:
        res: TFactors = TFactors([])
        for _, v in vars(self).items():
            if v is not None:
                factors = v.get_factors()
                res.extend(factors)
        return res

    def get_factors_from_factor_class(self, factor_class: TFactorClass) -> TFactors:
        cfg_fac = vars(self)[factor_class]
        sub_grp = [CFactor(cfg_fac.factor_class, factor_name) for factor_name in cfg_fac.factor_names]
        return sub_grp

    def get_mapper_name_to_class(self) -> dict[TFactorName, TFactorClass]:
        factors = self.get_factors()
        d = {f.factor_name: f.factor_class for f in factors}
        return d

    def get_factor_from_names(self, factor_names: TFactorNames) -> TFactors:
        mapper = self.get_mapper_name_to_class()
        res: TFactors = []
        for factor_name in factor_names:
            factor_class = mapper[factor_name]
            factor = CFactor(factor_class, factor_name)
            res.append(factor)
        return res


"""
--------------------------------------
Part III: Instruments and Universe
--------------------------------------
"""


@dataclass(frozen=True)
class CCfgInstru:
    sectorL0: str
    sectorL1: str


TInstruName = NewType("TInstruName", str)
TUniverse = NewType("TUniverse", dict[TInstruName, CCfgInstru])

"""
--------------------------------------
Part IV: Simulations
--------------------------------------
"""


@dataclass(frozen=True)
class CSimArgs:
    sim_id: str
    tgt_ret: CRet | None
    db_struct_sig: CDbStruct | None
    db_struct_ret: CDbStruct | None
    cost: float | None


TSimGrpIdByFacAgg = tuple[TFactorClass, TRetPrc]

"""
--------------------------------
Part V: models
--------------------------------
"""

TUniqueId = NewType("TUniqueId", str)


@dataclass(frozen=True)
class CModel:
    model_type: Literal["Ridge", "LGBM", "XGB"]
    model_args: dict

    @property
    def desc(self) -> str:
        return f"{self.model_type}"


@dataclass(frozen=True)
class CTestMdl:
    unique_Id: TUniqueId
    ret: CRet
    factors: TFactors
    trn_win: int
    model: CModel

    @property
    def layers(self) -> list[str]:
        return [
            self.unique_Id,  # M0005
            self.ret.ret_name,  # ClsRtn001L1
            f"W{self.trn_win:03d}",  # W060
            self.model.desc,  # Ridge
        ]

    @property
    def save_tag_mdl(self) -> str:
        return ".".join(self.layers)


"""
--------------------------------
Part VI: generic and project
--------------------------------
"""


@dataclass(frozen=True)
class CCfgAvlbUnvrs:
    win: int
    amount_threshold: float


@dataclass(frozen=True)
class CCfgTrn:
    wins: list[int]


@dataclass(frozen=True)
class CCfgPrd:
    wins: list[int]


@dataclass(frozen=True)
class CCfgSim:
    wins: list[int]


@dataclass(frozen=True)
class CCfgDecay:
    win: int
    rate: float


@dataclass(frozen=True)
class CCfgConst:
    COST: float
    COST_SUB: float
    SECTORS: list[str]
    LAG: int


@dataclass(frozen=True)
class CCfgProj:
    # --- shared
    calendar_path: str
    root_dir: str
    db_struct_path: str
    alternative_dir: str
    market_index_path: str
    by_instru_pos_dir: str
    by_instru_pre_dir: str
    by_instru_min_dir: str

    # --- project
    project_root_dir: str
    available_dir: str
    market_dir: str
    test_return_dir: str
    factors_by_instru_dir: str
    factors_aggr_avlb_dir: str
    sig_frm_fac_agg_dir: str
    sim_frm_fac_agg_dir: str
    evl_frm_fac_agg_dir: str
    opt_frm_slc_fac_dir: str
    sig_frm_fac_opt_dir: str
    sim_frm_fac_opt_dir: str
    evl_frm_fac_opt_dir: str
    mclrn_dir: str
    mclrn_cfg_file: str
    mclrn_mdl_dir: str
    mclrn_prd_dir: str
    sig_frm_mdl_prd_dir: str
    sim_frm_mdl_prd_dir: str
    evl_frm_mdl_prd_dir: str

    # --- project parameters
    universe: TUniverse
    avlb_unvrs: CCfgAvlbUnvrs
    mkt_idxes: dict
    const: CCfgConst
    trn: CCfgTrn
    prd: CCfgPrd
    sim: CCfgSim
    decay: CCfgDecay
    optimize: dict
    factors: dict
    selected_factors_pool: list
    cv: int
    mclrn: dict[str, dict]
    omega: dict

    @property
    def test_rets_wins(self) -> list[int]:
        return self.sim.wins + self.prd.wins

    def get_test_rets(self) -> TRets:
        res: TRets = []
        for win in self.sim.wins:
            ret_opn = CRet(ret_prc=TRetPrc("Opn"), win=win, lag=self.const.LAG)
            ret_cls = CRet(ret_prc=TRetPrc("Cls"), win=win, lag=self.const.LAG)
            res.append(ret_opn)
            res.append(ret_cls)
        return res


@dataclass(frozen=True)
class CCfgDbStruct:
    # --- shared database
    macro: CDbStruct
    forex: CDbStruct
    fmd: CDbStruct
    position: CDbStruct
    basis: CDbStruct
    stock: CDbStruct
    preprocess: CDbStruct
    minute_bar: CDbStruct

    # --- project database
    available: CDbStruct
    market: CDbStruct


TPid = NewType("TPid", str)
