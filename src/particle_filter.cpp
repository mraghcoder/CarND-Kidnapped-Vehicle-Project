/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

//function to normalize a vector:
vector<double> normalize_weights(vector<double> inputVector){

	//declare sum:
	double sum = 0.0;

	//declare and resize output vector:
	vector<double> outputVector ;
	outputVector.resize(inputVector.size());

	//estimate the sum:
	for (unsigned int i = 0; i < inputVector.size(); ++i) {
		sum += inputVector[i];
	}

	//normalize with sum:
	for (unsigned int i = 0; i < inputVector.size(); ++i) {
		outputVector[i] = inputVector[i]/sum ;
	}

	//return normalized vector:
	return outputVector ;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	random_device rd;
	mt19937 gen(rd());

	num_particles = 500;
	weights.resize(num_particles);
	particles.resize(num_particles);

	normal_distribution<double> x_init(0, std[0]);
	normal_distribution<double> y_init(0, std[1]);
	normal_distribution<double> theta_init(0, std[2]);

	for(int i=0; i<num_particles; i++){
		particles[i].id = i;
		particles[i].x = x + x_init(gen);
		particles[i].y = y + y_init(gen);
		particles[i].theta = theta + theta_init(gen);
		particles[i].weight = 1.0f;
		weights[i] = 1.0f;

	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	random_device rd;
	mt19937 gen(rd());

	normal_distribution<double> x_dist(0, std_pos[0]);
	normal_distribution<double> y_dist(0, std_pos[1]);
	normal_distribution<double> theta_dist(0, std_pos[2]);
	for(int i=0; i<num_particles; i++){
		double x = particles[i].x;
		double y = particles[i].y;
		double th = particles[i].theta;
		if(fabs(yaw_rate) > 0.001){
			particles[i].x = x + velocity/yaw_rate*(sin(th + yaw_rate*delta_t) - sin(th)) + x_dist(gen);
			particles[i].y = y + velocity/yaw_rate*(cos(th) - cos(th + yaw_rate*delta_t)) + y_dist(gen);
			particles[i].theta = th + yaw_rate*delta_t + theta_dist(gen);
		}
		else{
			particles[i].x = x + velocity*delta_t*cos(th) + x_dist(gen);
			particles[i].y = y + velocity*delta_t*sin(th) + y_dist(gen);
			particles[i].theta = th + theta_dist(gen);
		}

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0; i<observations.size(); i++){
		float min_dist = 1000;
		int min_idx = -1;
		for(int j=0; j<predicted.size(); j++){
			float distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[i].y) ;
			if (distance < min_dist){
				min_dist = distance;
				min_idx = j;
			}
		}
		//observations[i] = predicted[min_idx];
		observations[i].id = min_idx;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	const double bnd_den = 2*M_PI*sig_x*sig_y; //Denominator in Bivariate normal dist

	for(int pi=0; pi<num_particles; pi++){

		std::vector<LandmarkObs> observations_map; // Observations in map co-ordinates
		observations_map.resize(observations.size());
		double x_p = particles[pi].x;
		double y_p = particles[pi].y;
		double theta = particles[pi].theta;
		// Transformation of observations from vehicle co-ordinates to Map co-ordinates for each particle
		for (int i=0; i<observations.size(); i++){
			observations_map[i].x = x_p + observations[i].x*cos(theta) - observations[i].y*sin(theta);
			observations_map[i].y = y_p + observations[i].x*sin(theta) + observations[i].y*cos(theta);

		}
		std::vector<LandmarkObs> pred_measurements; // Predicted measurements
		for (int i=0; i< map_landmarks.landmark_list.size(); i++){
			// Dist from particle to map landmarks
			float dist_lm = dist(map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f, x_p, y_p);
			if (dist_lm <= sensor_range){

				LandmarkObs pred;
				pred.x = map_landmarks.landmark_list[i].x_f;
				pred.y = map_landmarks.landmark_list[i].y_f;
				pred.id = map_landmarks.landmark_list[i].id_i;
				pred_measurements.push_back(pred);
			}
		}
		// Data association
		dataAssociation(pred_measurements, observations_map);

		// Update weights
		double prob = 1.0;
		for(int i=0; i<observations_map.size(); i++){
			LandmarkObs pred = pred_measurements[observations_map[i].id];
			double x = observations_map[i].x;
			double y = observations_map[i].y;
			double mu_x = pred.x;
			double mu_y = pred.y;
			// Bivariate normal dist
			double num = exp(-0.5*(x-mu_x)*(x-mu_x)/(sig_x*sig_x)-0.5*(y-mu_y)*(y-mu_y)/(sig_y*sig_y));
			prob *= num/bnd_den;
		}
		weights[pi] = prob;
		particles[pi].weight = prob;
	}// particles
	//weights = normalize_weights(weights);
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device rd_wt;
	mt19937 gen_wts(rd_wt());
	discrete_distribution<int> wt_dist(weights.begin(), weights.end());
	std::vector<Particle> resamp_particles;
	for (int i=0; i<num_particles; i++){
		Particle p = particles[wt_dist(rd_wt)];
		resamp_particles.push_back(p);
	}
	particles = resamp_particles;

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
