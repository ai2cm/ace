from .constraint import Constraint
from .continuous import (
    PointwiseConstraint,
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    VariationalConstraint,
    VariationalDomainConstraint,
)
from .discrete import (
    SupervisedGridConstraint,
    DeepONetConstraint_Data,
    DeepONetConstraint_Physics,
)
