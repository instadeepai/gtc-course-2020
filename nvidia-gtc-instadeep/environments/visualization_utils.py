import matplotlib.animation
import matplotlib.pyplot as plt
from IPython.display import HTML


def render_env(env_renderer, show=False, frames=False, show_observations=False):
    """
    Renders the current state of the environment
    """
    env_renderer.render_env(show=show, frames=frames, show_observations=show_observations)
    image = env_renderer.gl.get_image()
    plt.figure(figsize=(image.shape[1] / 72.0, image.shape[0] / 72.0), dpi = 72)
    plt.axis("off")
    plt.imshow(image)
    plt.show()

def get_patch(frames):
    patch = plt.imshow(frames[0], alpha=0)
    plt.axis("off")
    plt.show()
    return patch

def animate_env(frames):
    """
    Plays an animation of rollouts
    """
    fig = plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = get_patch(frames)
    plt.axis('off')
    patch.set_alpha(1)
    animate = lambda i: patch.set_data(frames[i])
    anim = matplotlib.animation.FuncAnimation(fig, animate, frames=len(frames), interval=150)
    return HTML(anim.to_jshtml())
