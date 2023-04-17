"""
PDE configs for here

Architecture params need to be updated to more primative focused
This file is largely a place folder right now
"""

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, SI, II
from typing import Any, Union, List, Dict


@dataclass
class PDEConf:
    _target_: str = MISSING


@dataclass
class AdvectionDiffusionConf(PDEConf):
    _target_: str = "modulus.eq.pdes.advection_diffusion.AdvectionDiffusion"
    T: str = "T"
    D: str = "D"
    Q: int = 0
    rho: str = "rho"
    dim: int = 3
    time: bool = False


@dataclass
class DiffusionConf(PDEConf):
    _target_: str = "modulus.eq.pdes.diffusion.Diffusion"
    T: str = "T"
    D: str = "D"
    Q: int = 0
    dim: int = 3
    time: bool = False


@dataclass
class MaxwellFreqRealConf(PDEConf):
    _target_: str = "modulus.eq.pdes.electromagnetic.MaxwellFreqReal"
    ux: str = "ux"
    uy: str = "uy"
    uz: str = "uz"
    k: float = 1.0


@dataclass
class EnergyFluidConf(PDEConf):
    _target_: str = "modulus.eq.pdes.energy_equation.EnergyFluid"
    cp: str = "cp"
    kappa: str = "kappa"
    rho: str = "rho"
    nu: str = "nu"
    visc_heating: bool = False
    dim: int = 3
    time: bool = False


@dataclass
class LinearElasticityConf(PDEConf):
    _target_: str = "modulus.eq.pdes.linear_elasticity.LinearElasticity"
    E = 10
    nu = 0.3
    lambda_ = None
    mu = None
    rho: int = 1
    dim: int = 3
    time: bool = False


@dataclass
class LinearElasticityPlaneConf(PDEConf):
    _target_: str = "modulus.eq.pdes.linear_elasticity.LinearElasticityPlaneStress"
    E = 10
    nu = 0.3
    lambda_ = None
    mu = None
    rho: int = 1
    time: bool = False


@dataclass
class NavierStokesConf(PDEConf):
    _target_: str = "modulus.eq.pdes.navier_stokes.NavierStokes"
    nu = MISSING
    rho: float = 1
    dim: int = 3
    time: bool = True


@dataclass
class ZeroEquationConf(PDEConf):
    _target_: str = "modulus.eq.pdes.turbulence_zero_eq.ZeroEquation"
    nu = MISSING
    rho: float = 1
    dim: int = 3
    time: bool = True


@dataclass
class WaveEquationConf(PDEConf):
    _target_: str = "modulus.eq.pdes.wave_equation.WaveEquation"
    u = "u"
    c = "c"
    dim: int = 3
    time: bool = True


@dataclass
class HelmholtzEquationConf(PDEConf):
    _target_: str = "modulus.eq.pdes.wave_equation.HelmholtzEquation"
    u = MISSING
    K = MISSING
    dim: int = 3


def register_pde_configs() -> None:
    # TODO: Allow multiple pdes via multiple config groups
    # https://hydra.cc/docs/next/patterns/select_multiple_configs_from_config_group/
    cs = ConfigStore.instance()
    cs.store(
        group="pde",
        name="advection-diffusion",
        node=AdvectionDiffusionConf,
    )

    cs.store(
        group="pde",
        name="diffusion",
        node=DiffusionConf,
    )

    cs.store(
        group="pde",
        name="maxwell-real",
        node=MaxwellFreqRealConf,
    )

    cs.store(
        group="pde",
        name="energy-fluid",
        node=EnergyFluidConf,
    )

    cs.store(
        group="pde",
        name="linear-elasticity",
        node=LinearElasticityConf,
    )

    cs.store(
        group="pde",
        name="linear-elasticity-plane",
        node=LinearElasticityPlaneConf,
    )

    cs.store(
        group="pde",
        name="navier-stokes",
        node=NavierStokesConf,
    )

    cs.store(
        group="pde",
        name="zero-eq-turbulence",
        node=ZeroEquationConf,
    )

    cs.store(
        group="pde",
        name="wave",
        node=WaveEquationConf,
    )

    cs.store(
        group="pde",
        name="helmholtz",
        node=HelmholtzEquationConf,
    )
