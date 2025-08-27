
# 🧠 Perceptron Algorithm – Simple Explanation
**🌟 What is a Perceptron?**

- The perceptron is the simplest type of artificial neural network introduced by Frank Rosenblatt (1957).

- It is mainly used for binary classification (YES/NO, 0/1).

- Uses artificial neurons called Threshold Logic Units (TLU).


# 🏗 Components of a Perceptron

1. Inputs (features) → data values.

1. Weights → importance given to each input (learned during training).

1. Summation Function → adds up all (input × weight).

1. Bias (b) → helps shift the decision boundary.

1. Activation Function → step function, gives output as 0 or 1.

1. Output → final prediction (class 0 or 1).


# ⚙️ How It Works

1. Multiply each input by its weight.

1. Add them together and include the bias.

1. Pass the sum through the activation function:

    1. If sum ≥ threshold → Output = 1

    1. Else → Output = 0

1. Adjust weights & bias during training to minimize errors.



# 🧠 Learning Rule

**When prediction ≠ actual output:**

$$

w
new
	​

=w
old
	​

+η⋅(y−
y
^
	​

)⋅x
𝑏
𝑛
𝑒
𝑤
=
𝑏
𝑜
𝑙
𝑑
+
𝜂
⋅
(
𝑦
−
𝑦
^
)
b
new
	​

=b
old
	​

+η⋅(y−
y
^
	​

)

$$


- $$ η = learning rate. $$

- $$ y = true output, 𝑦^= predicted output.

$$



# ✅ Example

- If inputs are student’s marks and attendance → output = Pass (1) or Fail (0).
- The perceptron learns weights for "marks" and "attendance" and predicts new outcomes.


# 👉 In short:
**A Perceptron works like this:**

Inputs × Weights + Bias → Activation Function → Output


# 🔗 Perceptron Flow


```     Input Features (x1, x2, ..., xn)
                 │
                 ▼
        [ Multiply with Weights ]
                 │
                 ▼
      Weighted Sum ( Σ wi*xi + b )
                 │
                 ▼
        Activation Function (Step)
                 │
         ┌───────┴────────┐
         │                │
     Output = 1       Output = 0
   (Class Positive)  (Class Negative)

