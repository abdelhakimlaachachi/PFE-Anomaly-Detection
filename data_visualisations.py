import numpy as np
import matplotlib.pyplot as plt
import os

def plot_movement_analysis(npy_file_path, output_image_name, is_anomaly=False):
    data = np.load(npy_file_path)
    
    velocity = np.diff(data, axis=0)
    
    movement_magnitude = np.linalg.norm(velocity, axis=1)
    
    plt.figure(figsize=(10, 5))
    
    line_color = '#e74c3c' if is_anomaly else '#2ecc71'
    title_text = "Analyse EDA : Détection d'Anomalie (Mouvement Brusque)" if is_anomaly else "Analyse EDA : Mouvement Normal"
    
    plt.plot(movement_magnitude, color=line_color, linewidth=2.5, marker='o', markersize=4)
    
    plt.title(title_text, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Temps (Frames)", fontsize=12)
    plt.ylabel("Magnitude du Mouvement (Vélocité)", fontsize=12)
    
    plt.axhline(y=np.mean(movement_magnitude) + np.std(movement_magnitude)*2, 
                color='gray', linestyle='--', label='Seuil de variation (Threshold)')
    
    plt.fill_between(range(len(movement_magnitude)), movement_magnitude, color=line_color, alpha=0.2)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_image_name, dpi=300)
    plt.show()
    print(f"✅ Graphique sauvegardé sous : {output_image_name}")


plot_movement_analysis("C:/Users/hp/Jupyter/PFE/Final_version/dataset_pose/normal/0.npy", "mouvement_normal.png", is_anomaly=False)
plot_movement_analysis("C:/Users/hp/Jupyter/PFE/Final_version/dataset_pose/anomalie/0.npy", "mouvement_anomalie.png", is_anomaly=True)