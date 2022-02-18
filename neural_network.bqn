⟨Dual⟩ ← •Import "dual.bqn"
⟨GnuPlot⟩ ← •Import "BQN-Gnuplot/Gnuplot.bqn"
rng ← •MakeRand 2

# Two layer neural network with weights and biases. Both layers are sigmoid activated
Network ← {lr 𝕊 insz‿hidsz‿outsz:
    ⟨insz, outsz, lr⟩ ⇐

    real_l1_weights ← hidsz‿insz rng.Range 0
    real_l1_biases ← hidsz rng.Range 0

    real_l2_weights ← outsz‿hidsz rng.Range 0
    real_l2_biases ← outsz rng.Range 0

    n_params ⇐ (hidsz×insz) + hidsz + (outsz×hidsz) + outsz

    z ⇐ 0 Dual n_params⥊0

    l1_weights ⇐ (↕≢real_l1_weights) {h‿i 𝕊 w:
        idx ← (h × insz) + i
        eps ← 1⌾(idx⊸⊑) n_params⥊0
        w Dual eps
    }¨real_l1_weights

    l1_biases ⇐ (↕≢real_l1_biases) {h 𝕊 w:
        idx ← (hidsz × insz) + h
        eps ← 1⌾(idx⊸⊑) n_params⥊0
        w Dual eps
    }¨real_l1_biases

    l2_weights ⇐ (↕≢real_l2_weights) {o‿h 𝕊 w:
        idx ← (hidsz × insz) + hidsz + (o × hidsz) + h
        eps ← 1⌾(idx⊸⊑) n_params⥊0
        w Dual eps
    }¨real_l2_weights

    l2_biases ⇐ (↕≢real_l2_biases) {o 𝕊 w:
        idx ← (hidsz × insz) + hidsz + (outsz × hidsz) + o
        eps ← 1⌾(idx⊸⊑) n_params⥊0
        w Dual eps
    }¨real_l2_biases


    Forward ⇐ {
        x ← 𝕩
        # x ↩ W1 * x
        x ↩ {
            elementwise ← 𝕩 {𝕨.Mul 𝕩}¨ x
            total ← z {𝕨.Add 𝕩}´ elementwise
            total
        }˘l1_weights

        # x ↩ x + b1
        x ↩ l1_biases { 𝕨.Add 𝕩 }¨ x

        # x ↩ sigmoid(x) = 1 ÷ (1 + exp(-x))
        x ↩ { ((𝕩.EExp@).Add 1).PowC ¯1}¨ x

        # x ↩ W2 * x
        x ↩ {
            elementwise ← 𝕩 {𝕨.Mul 𝕩}¨ x
            total ← z {𝕨.Add 𝕩}´ elementwise
            total
        }˘l2_weights

        # x ↩ x + b2
        x ↩ l2_biases { 𝕨.Add 𝕩 }¨ x

        # x ↩ sigmoid(x) = 1 ÷ (1 + exp(-x))
        x ↩ { ((𝕩.EExp@).Add 1).PowC ¯1}¨ x
        x
    }

    Objective ⇐ {real_y 𝕊 y_hat:
        se ← real_y {(𝕩.Sub 𝕨).PowC 2}¨ y_hat
        z {𝕨.Add 𝕩}´ net.z {𝕨.Add 𝕩}´˘ se
    }
    Update ⇐ {𝕊 loss:
        l1_weights ↩ (↕≢l1_weights) {h‿i 𝕊 w:
            idx ← (h × insz) + i
            w.Sub lr × idx ⊑ loss.eps
        }¨l1_weights

        l1_biases ↩ (↕≢l1_biases) {h 𝕊 w:
            idx ← (hidsz × insz) + h
            w.Sub lr × idx ⊑ loss.eps
        }¨l1_biases

        l2_weights ↩ (↕≢l2_weights) {o‿h 𝕊 w:
            idx ← (hidsz × insz) + hidsz + (o × hidsz) + h
            w.Sub lr × idx ⊑ loss.eps
        }¨l2_weights

        l2_biases ↩ (↕≢l2_biases) {o 𝕊 w:
            idx ← (hidsz × insz) + hidsz + (outsz × hidsz) + o
            w.Sub lr × idx ⊑ loss.eps
        }¨l2_biases
    }
}

# # objective function:
# # f(x, y, z) = ⟨x > y, y < z⟩

# train_data ← ⟨
#     ⟨⟨1, 2, 3⟩, ⟨0, 1⟩⟩,
#     ⟨⟨5, ¯1, 10⟩, ⟨1, 1⟩⟩,
#     ⟨⟨2, 7, ¯3⟩, ⟨0, 0⟩⟩,
#     ⟨⟨20, 10, 0⟩, ⟨1, 0⟩⟩,
# ⟩

# f(x, y, z) = ⟨x ^ y, y | z⟩

train_data ← ⟨
    ⟨⟨0, 0, 0⟩, ⟨0, 0⟩⟩,
    ⟨⟨0, 0, 1⟩, ⟨0, 1⟩⟩,
    ⟨⟨0, 1, 0⟩, ⟨1, 1⟩⟩,
    ⟨⟨0, 1, 1⟩, ⟨1, 1⟩⟩,
    ⟨⟨1, 0, 0⟩, ⟨1, 0⟩⟩,
    ⟨⟨1, 0, 1⟩, ⟨1, 1⟩⟩,
    ⟨⟨1, 1, 0⟩, ⟨0, 1⟩⟩,
    ⟨⟨1, 1, 1⟩, ⟨0, 1⟩⟩,
⟩
train_x ← > {F x‿y: x}¨ train_data
train_y ← > {F x‿y: y}¨ train_data

net ← 0.1 Network 3‿5‿2

loss_history ← ↕0

Show ← {𝕤
    •Out "x ="
    •Show train_x

    y_hat ← {net.Forward 𝕩}˘ train_x

    •Out "y_hat ="
    •Show {𝕩.n}¨ y_hat
    •Out "y ="
    •Show train_y

    loss ← train_y net.Objective y_hat

    •Out "loss = " ∾ •Fmt loss.n
}

Step ← {𝕊 loud:
    y_hat ← {net.Forward 𝕩}˘ train_x

    loss ← train_y net.Objective y_hat

    net.Update loss

    loss_history ↩ loss_history ∾ ⟨loss.n⟩
}

Show 1
Step⍟2000 0
Show 1

# plot loss history

plt ← GnuPlot ⟨
    "title 'Neural network training'"
    "xlabel 'Epochs'"
    "ylabel 'Loss'"
⟩

plt.Plot loss_history‿"with lines"
plt.Show @
