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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = 20;
	
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x, std_x);
	
	// TODO: Create normal distributions for y and psi.
	normal_distribution<double> dist_y(y, std_y);
	
	normal_distribution<double> dist_theta(theta, std_theta);
	
	for (int i = 0; i < num_particles; ++i) 
	{
		Particle p;
		
		p.id = i;
		
		// TODO: Sample  and from these normal distrubtions like this: 
		//	 sample_x = dist_x(gen);
		//	 where "gen" is the random engine initialized earlier.
		
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		weights.push_back(p.weight);
		
		particles.push_back(p);
	}
	
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	for (int i = 0; i < num_particles; ++i) 
	{
		double new_x;
		double new_y;
		double new_theta;
		
		if (yaw_rate == 0) 
		{
			new_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			new_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			new_theta = particles[i].theta;
		}
		else
		{
			new_x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta));
			new_y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t)));
			new_theta = particles[i].theta + yaw_rate*delta_t;
		}
		
		normal_distribution<double> N_x(new_x, std_pos[0]);
		normal_distribution<double> N_y(new_y, std_pos[1]);
		normal_distribution<double> N_theta(new_theta, std_pos[2]);
		
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for(int i = 0; i < observations.size(); i++)
	{
		LandmarkObs obs = observations[i];
		double min_dist = std::numeric_limits<double>::max();
		
		int min_id = -1;
		
		for(int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];
			
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			
			if(distance < min_dist)
			{
				min_id = pred.id;
				min_dist = distance;
			}
		}
		
		observations[i].id = min_id;
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	weights.clear();
	
	for(int i = 0; i < num_particles; i++)
	{
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		
		vector<LandmarkObs> predicted_landmarks;
		
		for(int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{
			LandmarkObs lm_pred;
			
			lm_pred.id = map_landmarks.landmark_list[j].id_i;
			lm_pred.x = map_landmarks.landmark_list[j].x_f;
			lm_pred.y = map_landmarks.landmark_list[j].y_f;
			
			float dist_x = x - lm_pred.x;
			float dist_y = y - lm_pred.y;
			
			if((fabs(dist_x) <= sensor_range) && (fabs(dist_y) <= sensor_range))
			{
				predicted_landmarks.push_back(lm_pred);
			}
		}
			
		vector<LandmarkObs> transformed_observations;
		
		for(int j = 0; j < observations.size(); j++)
		{
			LandmarkObs transformed;
			
			transformed.x = cos(theta) * observations[j].x - sin(theta) * observations[j].y + x;
			transformed.y = sin(theta) * observations[j].x + cos(theta) * observations[j].y + y;
			transformed_observations.push_back(transformed);
		}
			
		dataAssociation(predicted_landmarks, transformed_observations);
		
		// Inspired in https://github.com/jeremy-shannon/CarND-Kidnapped-Vehicle-Project/blob/master/src/particle_filter.cpp
		
		for(int j = 0; j < transformed_observations.size(); j++)
		{
			double pred_x, pred_y;
			int obs_id = transformed_observations[j].id;
			double obs_x = transformed_observations[j].x;
			double obs_y = transformed_observations[j].y;

			for (int k = 0; k < predicted_landmarks.size(); k++) 
			{
				if (predicted_landmarks[k].id == obs_id) 
				{
				  pred_x = predicted_landmarks[k].x;
				  pred_y = predicted_landmarks[k].y;
				}
			}

			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double obs_w = (1 / (2 * M_PI * s_x *s_y)) * exp(-(pow(pred_x - obs_x, 2) / (2 * pow(s_x, 2)) + (pow(pred_y - obs_y, 2) / (2 * pow(s_y, 2)))));

			particles[i].weight = obs_w;
		}
		
		weights.push_back(particles[i].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	vector<Particle> resample_particles;
	
	std::discrete_distribution<> distribution(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) 
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
