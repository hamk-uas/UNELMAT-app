import numpy as np
from scipy.stats import entropy
from sklearn.cluster import DBSCAN

class BlisterClassifier:
    def __init__(self):
        self.size_categories = {
            'tiny': (0, 100),       # we can adjust these and below numbers
            'small': (100, 250),
            'medium': (250, 500),
            'large': (500, 1000),
            'very_large': (1000, float('inf'))
        }
        
        self.grade_characteristics = {
            2: {'size_dist': {'tiny': 0.6, 'small': 0.3, 'medium': 0.1}}, # we can adjust these and below numbers also
            3: {'size_dist': {'tiny': 0.2, 'small': 0.5, 'medium': 0.3}},
            4: {'size_dist': {'small': 0.2, 'medium': 0.5, 'large': 0.3}},
            5: {'size_dist': {'medium': 0.2, 'large': 0.5, 'very_large': 0.3}}
        }

    def calculate_density(self, bboxes):
        """Calculate density based on actual detection area"""
        if len(bboxes) == 0:
            return 0.0
        
        # Find the actual area where blisters are detected
        x_coords = []
        y_coords = []
        blister_areas = 0
        
        for bbox in bboxes:
            x_coords.extend([bbox[0], bbox[2]])
            y_coords.extend([bbox[1], bbox[3]])
            # Calculate individual blister area
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            blister_areas += width * height
        
        # Calculate area containing all blisters
        detection_width = max(x_coords) - min(x_coords)
        detection_height = max(y_coords) - min(y_coords)
        detection_area = detection_width * detection_height if detection_width > 0 and detection_height > 0 else 1
        
        # Calculate density as percentage of area covered
        density = (blister_areas * 100) / detection_area
        return round(density, 2)  # Round to 2 decimal places

    def analyze_spatial_distribution(self, bboxes):
        if len(bboxes) < 2:
            return {'n_clusters': 0, 'avg_cluster_size': 0, 'density': 0}

        # Calculate centers for clustering
        centers = np.array([[
            (bbox[0] + bbox[2])/2,
            (bbox[1] + bbox[3])/2
        ] for bbox in bboxes])

        # Adaptive DBSCAN parameters based on number of detections
        if len(bboxes) > 50:
            eps = 70
            min_samples = 3
        else:
            eps = 50
            min_samples = 2

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        cluster_sizes = []
        for label in set(labels):
            if label != -1:
                cluster_size = np.sum(labels == label)
                cluster_sizes.append(cluster_size)

        density = self.calculate_density(bboxes)

        return {
            'n_clusters': n_clusters,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'density': density,
            'total_detections': len(bboxes)
        }

    def calculate_size(self, bbox):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height

    def categorize_size(self, size):
        for category, (min_size, max_size) in self.size_categories.items():
            if min_size <= size < max_size:
                return category
        return 'very_large'

    def generate_histogram(self, bboxes):
        sizes = [self.calculate_size(bbox[:4]) for bbox in bboxes]
        categories = [self.categorize_size(size) for size in sizes]
        
        hist = {category: 0 for category in self.size_categories.keys()}
        for category in categories:
            hist[category] += 1
            
        total = sum(hist.values())
        if total > 0:
            hist = {k: v/total for k, v in hist.items()}
            
        return hist

    def classify_image(self, bboxes):
        if len(bboxes) == 0:
            return {
                'grade': 'Unknown',
                'stage': 0,
                'confidence': 0,
                'histogram': {},
                'spatial_features': {'n_clusters': 0, 'avg_cluster_size': 0, 'density': 0}
            }

        hist = self.generate_histogram(bboxes)
        spatial_features = self.analyze_spatial_distribution(bboxes)
        
        grade_scores = {}
        for grade, characteristics in self.grade_characteristics.items():
            # Size distribution similarity
            size_similarity = self.compare_distributions(hist, characteristics['size_dist'])
            
            # Spatial score with density consideration
            spatial_score = self.calculate_spatial_score(spatial_features, grade)
            
            # Detection count factor
            count_factor = self.calculate_count_factor(len(bboxes), grade)
            
            # Weighted combination
            if len(bboxes) > 50:
                grade_scores[grade] = (0.4 * size_similarity + 
                                     0.4 * spatial_score + 
                                     0.2 * count_factor)
            else:
                grade_scores[grade] = (0.5 * size_similarity + 
                                     0.3 * spatial_score + 
                                     0.2 * count_factor)

        assigned_grade = max(grade_scores.items(), key=lambda x: x[1])[0]
        stage = self.determine_stage(hist, spatial_features, assigned_grade)
        
        return {
            'grade': assigned_grade,
            'stage': stage,
            'confidence': grade_scores[assigned_grade],
            'histogram': hist,
            'spatial_features': spatial_features
        }

    def calculate_count_factor(self, num_detections, grade):
        grade_ranges = {
            2: (1, 10),
            3: (5, 25),
            4: (20, 50),
            5: (40, 100)
        }
        
        min_count, max_count = grade_ranges[grade]
        if min_count <= num_detections <= max_count:
            return 1.0
        elif num_detections < min_count:
            return num_detections / min_count
        else:
            return max_count / num_detections

    def compare_distributions(self, hist1, hist2):
        categories = list(self.size_categories.keys())
        p = np.array([hist1.get(cat, 0) for cat in categories])
        q = np.array([hist2.get(cat, 0) for cat in categories])
        
        p = p + 1e-10
        q = q + 1e-10
        
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        m = (p + q) / 2
        js_div = (entropy(p, m) + entropy(q, m)) / 2
        
        return 1 / (1 + js_div)

    def calculate_spatial_score(self, spatial_features, grade):
        grade_spatial = {
            2: {'max_clusters': 3, 'max_cluster_size': 4, 'max_density': 15},
            3: {'max_clusters': 5, 'max_cluster_size': 6, 'max_density': 25},
            4: {'max_clusters': 8, 'max_cluster_size': 10, 'max_density': 35},
            5: {'max_clusters': 12, 'max_cluster_size': 15, 'max_density': 45}
        }
        
        expected = grade_spatial[grade]
        
        cluster_score = min(spatial_features['n_clusters'] / expected['max_clusters'], 1)
        size_score = min(spatial_features['avg_cluster_size'] / expected['max_cluster_size'], 1)
        density_score = min(spatial_features['density'] / expected['max_density'], 1)
        
        return (cluster_score + size_score + density_score) / 3

    def determine_stage(self, hist, spatial_features, grade):
        size_progression = sum(i * v for i, v in enumerate(hist.values()))
        spatial_progression = (
            spatial_features['n_clusters'] * 0.3 +
            spatial_features['avg_cluster_size'] * 0.3 +
            spatial_features['density'] * 0.4
        )
        
        total_progression = (size_progression + spatial_progression) / 2
        
        if total_progression < 0.3:
            return 2
        elif total_progression < 0.5:
            return 3
        elif total_progression < 0.7:
            return 4
        else:
            return 5