⟨Dual⟩ ←•Import "dual.bqn"

# maximize sinx + cos xy with gradient ascent
BlackBox ← {F x‿y:
    (x.Sin @).Add (x.Mul y.Add 2).Cos@
}

# initial conditions close to local minima
x ← ¯1.59 Dual 1‿0
y ← ¯0.03 Dual 0‿1

eps ← 0.1 # positive epsilon = gradient ascent

history ← ↕0

Step ← {F:
    r ← BlackBox x‿y
    history ↩ history ∾ 1 ⥊ r.n
    •Show "f ⟨x,y⟩ = f ⟨" ∾ (•Repr x.n) ∾ "," ∾ (•Repr y.n) ∾ "⟩ = " ∾ r.Disp@
    dx ← eps × 0 ⊑ r.eps
    dy ← eps × 1 ⊑ r.eps
    # •Show "⟨dx, dy⟩ = ⟨" ∾ (•Repr dx) ∾ ", " ∾ (•Repr dy) ∾ "⟩"
    x ↩ (x.n+dx) Dual x.eps
    y ↩ (y.n+dy) Dual y.eps
}

Step ⍟ 20@

# •Plot history
mi ← ⌊´history
ma ← ⌈´history
{ •Out "." ∾˜ " "⥊˜⌊0.5+40 × 𝕩 (÷○(-⟜mi)) ma }¨ history


