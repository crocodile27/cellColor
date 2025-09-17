import random
class ColorMixin:
    
    colors_rgb = {
        "Dartmouth Green": (0, 102, 44),      # Hex: #00662c
        "Spring Bud": (175, 247, 17),         # Hex: #aff711
        "Pear": (214, 237, 86),               # Hex: #d6ed56
        "Apple Green": (142, 166, 4),         # Hex: #8ea604
        "Gamboge": (236, 159, 5),             # Hex: #ec9f05
        "Rust": (191, 49, 0),                 # Hex: #bf3100
        "OU Crimson": (128, 15, 15),          # Hex: #800f0f
        "Citrine": (229, 209, 44),            # Hex: #e5d12c
        "Vanilla": (255, 236, 159),           # Hex: #ffec9f
        "Sunglow": (255, 202, 97),            # Hex: #ffca61
        "Bittersweet": (248, 118, 92),        # Hex: #f8765c
        "Melon": (255, 192, 183),             # Hex: #ffc0b7
        "Indian Red": (192, 104, 105),        # Hex: #c06869
        "Folly": (255, 66, 85),               # Hex: #ff4255
        "Red": (255, 0, 0)
    }

    def generate_unique_color(self):
            available_colors = [
                value for value in self.colors_rgb.values()
                if value not in self.selected_genes.values()
            ]
            if not available_colors:
                return random.choice(list(self.colors_rgb.values()))
            return random.choice(available_colors)
        
    def generate_unique_cluster_color(self):
            available_cluster_colors = [
                value for value in self.colors_rgb.values()
                if value not in self.selected_clusters.values()
            ]
            if not available_cluster_colors:
                return random.choice(list(self.colors_rgb.values()))
            return random.choice(available_cluster_colors)
