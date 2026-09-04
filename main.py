import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time


x = jnp.linspace(-2.0, 2.0, 201)
a = 1.0
b = 1.0
c = 1.0
rate = 0.01
steps = 1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))


equation = lambda x: (
    2 * x ** 2 + -7/5 * x + 3
)

key = jax.random.key(0)
noise = 0.5 * jax.random.normal(key, shape=x.shape)
y = equation(x) + noise

x_train = x[::2]
y_train = y[::2]

x_test = x[1::2]
y_test = y[1::2]

model = lambda a, b, c, x: (
    (a * x ** 2) + (b * x) + c
)


def lossfunc(a, b, c, x, y):
    guess = model(a, b, c, x)
    loss = jnp.subtract(guess, y)
    return jnp.mean(jnp.square(loss))

loss_and_grad = jax.value_and_grad(
    lossfunc,
    argnums=(0, 1, 2)
)

def update(a, b, c, x, y):
    loss, gradients = loss_and_grad(a, b, c, x, y)
    grad_a, grad_b, grad_c = gradients
    a = a - jnp.multiply(rate, grad_a)
    b = b - jnp.multiply(rate, grad_b)
    c = c - jnp.multiply(rate, grad_c)
    print("Tracing update")

    return a, b, c, loss


def train(a, b, c, x_train, y_train):
    initial_state = (a, b, c)

    def loop_body(state, _):
        a, b, c = state
        new_a, new_b, new_c, loss = update(
            a, b, c, x_train, y_train
        )
        new_state = (new_a, new_b, new_c)
        return new_state, loss

    final_state, losses = jax.lax.scan(
        loop_body,
        initial_state,
        xs=None,
        length=steps
    )
    return final_state, losses

def main(a, b, c, x_train, y_train, x_all, y_all):
    compiled_train = jax.jit(train)
    final_state, losses = compiled_train(
        a, b, c, x_train, y_train
    )
    train_loss = lossfunc(
        a, b, c, x_train, y_train
    )

    test_loss = lossfunc(
        a, b, c, x_test, y_test
    )

    overall_loss = lossfunc(
        a, b, c, x_all, y_all
    )
    a, b, c = final_state
    print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print("Training loss:", train_loss)
    print("Testing loss:", test_loss)
    print("Overall loss:", overall_loss)
    predictions = model(a, b, c, x)
    ax1.scatter(x_train, y_train, label="Noisy observations")
    ax1.scatter(x_test, y_test, label="target data")
    ax1.plot(x, predictions, label="predictions")
    ax2.plot(losses)
    ax2.set_title("losses")
    plt.tight_layout()
    plt.show()

    



# 3. Display the live plot window


main(a, b, c, x_train, y_train, x, y)
