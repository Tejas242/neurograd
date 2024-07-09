#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include "neurograd.h"

class Neuron {
private:
    std::vector<Value> w;
    Value b;

public:
    Neuron(int nin) : b(Value((double)rand() / RAND_MAX * 2 - 1)) {
        for (int i = 0; i < nin; ++i) {
            w.push_back(Value((double)rand() / RAND_MAX * 2 - 1));
        }
    }

    Value operator()(const std::vector<Value>& x) {
        Value act = b;
        for (size_t i = 0; i < w.size(); ++i) {
            act = act + w[i] * x[i];
        }
        return act.tanh();
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> params;
        for (auto& weight : w) {
            params.push_back(&weight);
        }
        params.push_back(&b);
        return params;
    }
};

class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int nin, int nout) {
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(Neuron(nin));
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& x) {
        std::vector<Value> outs;
        for (auto& neuron : neurons) {
            outs.push_back(neuron(x));
        }
        return outs.size() == 1 ? std::vector<Value>{outs[0]} : outs;
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> params;
        for (auto& neuron : neurons) {
            auto neuron_params = neuron.parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }
};

class MLP {
private:
    std::vector<Layer> layers;

public:
    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());
        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.push_back(Layer(sz[i], sz[i+1]));
        }
    }

    std::vector<Value> operator()(std::vector<Value> x) {
        for (auto& layer : layers) {
            x = layer(x);
        }
        return x;
    }

    std::vector<Value*> parameters() {
        std::vector<Value*> params;
        for (auto& layer : layers) {
            auto layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};