import numpy as np

class OrthogonalDemonstrationEvaluator:
    def __init__(self):
        self.given_demonstrations = []
        self.desired_demonstrations = []

    def _reset_evaluator(self):
        self.given_demonstrations = []
        self.desired_demonstrations = []        

    def _orthogonal_projection(self, vector, space):
        if len(space) == 0:
            return vector
        else:
            basis = np.array(space).T
            # Compute pseudo-inverse with regularization
            pseudo_inverse = np.linalg.inv(basis.T @ basis + 1e-5 * np.eye(basis.shape[1])) @ basis.T
            coeff = pseudo_inverse @ vector
            projection = vector - basis @ coeff
            return projection

    def _normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def _recover_original_vector(self, normalized_vector):
        original_norm = 1 / normalized_vector[-1]
        original_vector = [v * original_norm for v in normalized_vector]
        return original_vector

    def generate_feedback(self, delta):
        feedback = []

        # Position feedback
        if delta[0] > 0:
            feedback.append("Increase X position.")
        elif delta[0] < 0:
            feedback.append("Decrease X position.")

        if delta[1] > 0:
            feedback.append("Increase Y position.")
        elif delta[1] < 0:
            feedback.append("Decrease Y position.")

        # Velocity feedback
        if delta[2] > 0:
            feedback.append("Increase X velocity.")
        elif delta[2] < 0:
            feedback.append("Decrease X velocity.")

        if delta[3] > 0:
            feedback.append("Increase Y velocity.")
        elif delta[3] < 0:
            feedback.append("Decrease Y velocity.")

        return feedback

    def add_demonstration(self, demo_vector):
        self.given_demonstrations.append(demo_vector)

        if len(self.desired_demonstrations) == 0:
            self.desired_demonstrations.append(demo_vector)
            return 0, []

        projected_vector = self._orthogonal_projection(demo_vector, self.desired_demonstrations)
        normalized_vector = self._normalize_vector(projected_vector)
        desired_vector = self._recover_original_vector(normalized_vector)

        self.desired_demonstrations.append(desired_vector)

        difference = np.linalg.norm(demo_vector - desired_vector)
        delta = desired_vector - demo_vector

        feedback = self.generate_feedback(delta)

        return difference, feedback

    def evaluate_demonstrations(self):
        if len(self.given_demonstrations) == 0:
            return "No demonstrations provided yet.", []

        quality_scores = []
        feedback_list = []

        demos_copy = self.given_demonstrations.copy()  # Create a copy to iterate

        for demo in demos_copy:
            score, feedback = self.add_demonstration(demo)
            quality_scores.append(score)
            feedback_list.append(feedback)

        average_quality = np.mean(quality_scores)

        return average_quality, feedback_list