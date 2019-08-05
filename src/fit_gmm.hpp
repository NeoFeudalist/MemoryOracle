#pragma once

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

// Helper function for indexing vectors as matrices.
double mat_index(double row, double col, double cols) {
	return (row * cols) + col;
}
/* Apply one iteration of expectation-maximization on a GaussianMM.
See https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf.
	
We may have wanted to use regularization or a Bayesian prior, but in practice 
there will be many data points (thousands in one hour!). Per the 
Bernstein-Von Mises theorem, the prior becomes less influential with more 
data. */
void gmm_em_step(GaussianMM& gmm, std::vector<double> obs, double eps=0.0001) {
	std::vector<double> weights;
	double gmm_pdf;

	for (double x : obs) {
		gmm_pdf = gmm.pdf(x);
		for (GaussianMMComponent dist : gmm.dists) {
			weights.push_back(dist.scaled_pdf(x) / gmm_pdf);
		}
	}

	int N = obs.size();
	int K = gmm.dists.size();

	for (int k = 0; k < K; k++) {
		double n = 0;
		double new_mean = 0;
		double new_sd = 0;

		for (int i = 0; i < N; i++) {
			double w_ik = weights[mat_index(i, k, K)];
			n += w_ik;
			new_mean += w_ik * obs[i];
		}

		new_mean /= n;

		// Standard deviation has to be calculated after new mean.
		for (int i = 0; i < N; i++) {
			double w_ik = weights[mat_index(i, k, K)];
			new_sd += w_ik * (obs[i] - new_mean) * (obs[i] - new_mean);
		}

		new_sd /= n;

		// Update parameters.
		gmm.dists[k].weight = n / N;
		gmm.dists[k].dist.mean = new_mean;
		// To prevent standard deviation underflow.
		gmm.dists[k].dist.sd = std::max(eps, new_sd);
	}
}

