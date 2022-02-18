# Generalized dual numbers are the field R extended with n distinct new elements ε_0, ε_1, ..., such that ε_i * ε_j = 0
# using MAGIC one can derive that f(x + a ε_0, y + b ε_1, ...) = f(x, y, ...) + a ε_0 f_0(x, y, ...) + b ε_1 f_1(x, y, ...) where f_n is the derivatie of f with respect to the nth argument
# thus it can be used for FULLY AUTOMATIC DIFFERENTIATION (real)

# Functions pre- or suffixed by C expets a real number as argument, as opposed to a Dual number.
Dual ⇐ {n F eps:
    eps‿n ⇐
    # displays the number in form n + a ε_0 + b ε_1 + ...
    Disp ⇐ {𝕤
        coeffs ← ∾ (eps≠0) / (" + "⊸∾)¨ (•Repr¨ eps) ∾¨ ("ε_"⊸∾)¨ •Repr¨ ↕≠eps
        (•Repr n) ∾ coeffs
    }

    # does 𝕩 have the same dimensions as this?
    DimCheck ← {
    𝕊 n : 6 = •Type 𝕩?
        "must be equal in dimension" ! 𝕩.eps (=○≠) eps
        𝕩
    ;
        𝕩 Dual 0¨eps
    }
    C ⇐ { 𝕩 Dual 0×eps }

    Add ⇐ {
        𝕩 ↩ DimCheck 𝕩
        (n+𝕩.n) Dual (eps+𝕩.eps)
    }

    Sub ⇐ {
        𝕩 ↩ DimCheck 𝕩
        (n-𝕩.n) Dual (eps-𝕩.eps)
    }

    Mul ⇐ {
        𝕩 ↩ DimCheck 𝕩
        (n×𝕩.n) Dual (n×𝕩.eps) + (𝕩.n×eps)
    }

    PowC ⇐ {
    F 0:
        C 1 # special case for x⋆0
    ;
        (n⋆𝕩) Dual eps × 𝕩×n⋆𝕩-1
    }

    EExp ⇐ {F:
        expn ← ⋆n
        expn Dual eps × expn
    }


    CExp ⇐ {F:
        (𝕩⋆n) Dual eps × (⋆⁼𝕩)×𝕩⋆n
    }

    Sin ⇐ {F:
        (•math.Sin n) Dual eps × •math.Cos n
    }

    Cos ⇐ {F:
        (•math.Cos n) Dual -eps × •math.Sin n
    }
}
