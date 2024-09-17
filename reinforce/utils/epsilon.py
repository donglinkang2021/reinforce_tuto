import math

class Epsilon:
    """Greedy Epsilon policy"""
    def __init__(
            self, 
            start:float = 1,
            end:float = 0,
            decay:int = 1000,
        ) -> None:
        self.value = start
        self.start = start
        self.end = end
        self.decay = decay
        self.sample_count = 0

    def update(self) -> float:
        self.sample_count += 1
        temperature = math.exp(-1. * self.sample_count / self.decay)
        self.value = self.end + (self.start - self.end) * temperature
        return self.value
    
    def reset(self) -> float:
        self.sample_count = 0
        self.value = self.start
        return self.value
    
    def __call__(self) -> float:
        return self.value
    
    def __repr__(self) -> str:
        return f"Epsilon(start={self.start}, end={self.end}, decay={self.decay})"
    
    def __float__(self) -> float:
        return self.value
    
if __name__ == "__main__":

    settings = [
        (0.95, 0.01, 100),
        (1, 0.1, 1000),
        (1, 0.01, 2000),
        (1, 0.001, 5000),
    ]

    epsilon_list = [Epsilon(start, end, decay) for start, end, decay in settings]

    import matplotlib.pyplot as plt
    for epsilon in epsilon_list:
        values = []
        for _ in range(5000):
            values.append(epsilon())
            epsilon.update()
        plt.plot(values, label=str(epsilon))

    plt.title('Greedy Epsilon Policy')
    plt.xlabel('Sample Count')
    plt.ylabel('Epsilon Value')
    plt.legend()
    plt.show()

# python -m reinforce.utils.epsilon