⟨Dual⟩ ← •Import "dual.bqn"
⟨GnuPlot⟩ ← •Import "BQN-Gnuplot/Gnuplot.bqn"
rng ← •MakeRand 0

lr ← 0.01

# Single layer neural network with weights, biases and a sigmoid activation function
Network ← {insz 𝕊 outsz:
    ⟨insz, outsz⟩ ⇐

    real_weights ← outsz‿insz rng.Range 0
    real_biases ← outsz rng.Range 0

    n_params ⇐ (outsz×insz) + outsz

    z ⇐ 0 Dual n_params⥊0

    weights ⇐ (↕≢real_weights) {o‿i 𝕊 w:
        idx ← (o × insz) + i
        eps ← 1⌾(idx⊸⊑) n_params⥊0
        w Dual eps
    }¨real_weights

    biases ⇐ (↕≢real_biases) {o 𝕊 w:
        idx ← (outsz × insz) + o
        eps ← 1⌾(idx⊸⊑) n_params⥊0
        w Dual eps
    }¨real_biases

    Forward ⇐ {
        x ← 𝕩
        # x ↩ W * x
        x ↩ {
            elementwise ← 𝕩 {𝕨.Mul 𝕩}¨ x
            total ← z {𝕨.Add 𝕩}´ elementwise
            total
        }˘weights

        # x ↩ x + b
        x ↩ biases {
            𝕨.Add 𝕩
        }¨ x
        # x ↩ sigmoid(x) = 1 ÷ (1 + exp(-x))
        x ↩ { ((𝕩.EExp@).Add 1).PowC ¯1}¨ x
    }

    Objective ⇐ {real_y 𝕊 y_hat:
        se ← real_y {(𝕩.Sub 𝕨).PowC 2}¨ y_hat
        z {𝕨.Add 𝕩}´ net.z {𝕨.Add 𝕩}´˘ se
    }
    Update ⇐ {𝕊 loss:
        weights ↩ (↕≢weights) {o‿i 𝕊 w:
            idx ← (o × insz) + i
            w.Sub lr × idx ⊑ loss.eps
        }¨weights

        biases ↩ (↕≢biases) {o 𝕊 w:
            idx ← (outsz × insz) + o
            w.Sub lr × idx ⊑ loss.eps
        }¨biases
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

# f(x, y, z) = ⟨~x | y, y | z⟩

train_data ← ⟨
    ⟨⟨0, 0, 0⟩, ⟨1, 0⟩⟩,
    ⟨⟨0, 0, 1⟩, ⟨1, 1⟩⟩,
    ⟨⟨0, 1, 0⟩, ⟨1, 1⟩⟩,
    ⟨⟨0, 1, 1⟩, ⟨1, 1⟩⟩,
    ⟨⟨1, 0, 0⟩, ⟨0, 0⟩⟩,
    ⟨⟨1, 0, 1⟩, ⟨0, 1⟩⟩,
    ⟨⟨1, 1, 0⟩, ⟨1, 1⟩⟩,
    ⟨⟨1, 1, 1⟩, ⟨1, 1⟩⟩,
⟩
train_x ← > {F x‿y: x}¨ train_data
train_y ← > {F x‿y: y}¨ train_data

net ← 3 Network 2

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

    •Out "loss = " ∾ •Fmt loss.n

    net.Update loss

    loss_history ↩ loss_history ∾ ⟨loss.n⟩
}

Show 1
Step⍟1000 0
Show 1

# plot loss history

plt ← GnuPlot ⟨
    "title 'Neural network training'"
    "xlabel 'Epochs'"
    "ylabel 'Loss'"
⟩

plt.Plot loss_history‿"with lines"
plt.Show @
