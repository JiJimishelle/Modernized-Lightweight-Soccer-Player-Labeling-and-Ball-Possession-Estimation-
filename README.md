# Modernized-Lightweight-Soccer-Player-Labeling-and-Ball-Possession-Estimation-


This diagram shows the overall framework of my re-implementation, inspired by Liu et al. (2009) but updated with modern deep-learning methods.
We start with the input video, which is processed along two parallel pipelines — one for players and one for the ball.
On the left side, we use YOLOv8 trained on the COCO dataset to detect all players in each frame.
The detected player regions are then processed through K-Means clustering, which groups them by team jersey colors in an unsupervised way.
Next, a Referee and Outlier Filtering step removes yellow jerseys and misclassified players to improve reliability.
After that, Norfair tracking with inertia voting assigns consistent IDs to each player, maintaining identity across frames even during occlusion or motion blur.
On the right side, another YOLOv8 model — trained with our custom ball dataset (best.pt) — detects the ball in each frame.
The outputs from both sides are combined in the ball possession estimation module, which identifies which team controls the ball based on the nearest detected player.
Finally, the system produces an output video that visualizes player IDs, team colors, and possession statistics in real-time.
Original 2009 Method	Re-Implementation Enhancement
Haar-based player detector	YOLOv8 (deep learning-based)
Hand-defined HSV thresholds	KMeans color clustering
Frame-by-frame labeling	Norfair tracker with inertia memory
Manual player labeling	Automatic clustering + adaptive thresholds
CPU-only slow tracking	Real-time compatible pipeline


Main goal of my re-implementation is to build a fully automated system that:
•	Detects players and the ball using deep learning 
•	Label player teams and referees automatically based on jersey colors
•	Tracks each player over time
•	Calculates which team has ball possession using spatial proximity
•	Keep it lightweight

**Player & Ball Detection (YOLOv8)**
The first stage uses YOLOv8n, a lightweight object detection model:
The player detection step uses the YOLOv8 model, pretrained on the COCO dataset.
It detects all visible players in each frame with high confidence, filtering out other people or objects. After detecting the players, the system crops each bounding box to isolate only the upper torso area, where the jersey color is dominant.
This region is crucial for the next step — team classification using color clustering.
In the ball detection, I prepared custom frames with a ball bounding box. And a custom-trained it with YOLOv8 model which is best.pt. It gas high precision 98% and recall 92%

**Kmeans Clustering**
Instead of manually defining HSV thresholds, the system applies unsupervised KMeans clustering:
•	Collects color features during a warm-up phase (first 40 frames).
•	Clusters into two groups (Team A, Team B).
•	Computes per-cluster thresholds to reject outliers (e.g., referees).
•	Smooths centroids over time using Exponential Moving Average (EMA) for stability.
This allows the model to adapt to lighting changes and color variations dynamically.

**Player Tracking (Norfair)**
To maintain consistent player IDs across frames, the system uses Norfair, a lightweight tracker:
•	Converts YOLO detections into Norfair Detection objects.
•	Uses Euclidean distance matching between consecutive frames.
•	Each tracked player is assigned a persistent ID.
To handle temporary detection noise, each track keeps a short inertia history window (20 frames) that votes for the most frequent recent label (Team A, Team B, etc.)


**Ball Possession Estimation**
The system computes the center of the detected ball and measures its distance to all players’ bounding box centers.
•	The nearest player determines the current possession team.
•	If detection is missing temporarily, possession is held for a short memory window (hold_ms = 800ms).
Possession counts are accumulated per team and visualized as a possession bar at the bottom of the frame.

**Results & Discussion**
•	The system successfully detects, labels, and tracks all players through unsupervised learning.
•	Each player maintains a consistent ID across frames.
•	Referees and outliers are handled properly.
•	Possession estimation reflects real ball control dynamics.
Limitations include:
•	Some early frames show Unknown labels during cluster warm-up.
•	Reflections or light variation can occasionally shift team clusters.
<img width="468" height="648" alt="image" src="https://github.com/user-attachments/assets/6011f93c-0073-4218-958a-72e4b1cf3725" />
