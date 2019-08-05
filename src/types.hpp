#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

/*
A header containing types used by the program. Type methods can also be put in 
this file.
 
-> Template restrictions (until Concepts become a common thing).
	- Dist: must have the methods:
		> double pdf(double x).
*/

struct MemoryPoint {
	unsigned long long memory_usage;
	unsigned long long swap_usage;
	std::chrono::steady_clock::time_point timestamp;

	unsigned long long total_usage() const {
		return memory_usage + swap_usage;
	}
};


// Normal (Gaussian) distribution.
struct NormalDist {
	double mean;
	double sd;

	double pdf(double x) {
		const static double PI = 3.14159265358979323846;	
		double factor1 = 1 / std::sqrt(2 * PI * sd * sd);
		double factor2 = std::exp(-std::pow(x - mean, 2) / (2 * sd * sd));
		return factor1 * factor2;
	}

	double log_pdf(double x) {
		const static double PI = 3.14159265358979323846;
		return -std::log(sd) - (std::log(2 * PI) / 2) - (0.5 * std::pow(((x - mean) / sd), 2));
	}
};

// A component of a mixture model, with distribution Dist.
template <typename Dist>
struct MMComponent {
	Dist dist;
	double weight;

	double scaled_pdf(double x) {
		return dist.pdf(x) * weight;
	}
};

// A mixture model of Dists.
template <typename Dist>
struct MM {
	std::vector< MMComponent<Dist> > dists;

	double pdf(double x) {
		std::vector<double> weights;
		std::vector<double> pdfs;
		// Must ideally be 1, but floating-point numbers are tricky.
		double total_weight;

		// Get individual mixture weights.
		std::transform(dists.begin(), dists.end(), std::back_inserter(weights),	
		[](MMComponent<Dist>& dist) -> double {return dist.weight;});
		// Sum mixture weights to get divisor.
		total_weight = std::accumulate(weights.begin(), weights.end(), 0.);
		std::transform(dists.begin(), dists.end(), std::back_inserter(pdfs),
		// Get probability densities for each component.
		[x](MMComponent<Dist>& dist) -> double {return dist.scaled_pdf(x);});

		// Return weighted average of probability densities.
		return std::accumulate(pdfs.begin(), pdfs.end(), 0.) / total_weight;
	}

	double log_pdf(double x) {
		return pdf(x);
	}
};

// The following two typedefs define a Gaussian mixture model, 
// or one where the components are Normal distributions.
typedef MMComponent<NormalDist> GaussianMMComponent;
typedef MM<NormalDist> GaussianMM;