using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace MNIST
{
    [Serializable]
    public class Layer
    {
        private static Random random = new Random(); //SEED?

        private Vector activation;
        private Vector biases;
        private Matrix weights;

        private Vector deriv_biases;
        private Matrix deriv_weights;

        public Vector Activation { get => activation; set => activation = value; }
        public Vector Biases { get => biases; set => biases = value; }
        public Matrix Weights { get => weights; set => weights = value; }

        public int Size => activation.Dimensions;
        public int Connections => biases.Dimensions;

        public Layer(int size, int connections)
        {
            this.Activation = new Vector(size);
            this.Biases = new Vector(connections);
            this.Weights = new Matrix(size, connections);

            this.biases.ForEach(x => (float)random.NextDouble() * 2 - 1);
            this.weights.ForEach(x => (float)random.NextDouble() * 2 - 1);
        }

        public Layer()
        {

        }

        public Vector GetOutput(Vector activation) => (Weights * (this.activation = activation) + biases).ForEach(MathUtils.Sigmoid);

        public Vector Backprop(Vector error)
        {
            error = (Weights * this.activation + biases).ForEach(MathUtils.DerivSigmoid) * error;
            Vector deriv_activation = new Vector(Weights.Width);
            Matrix deriv_weights = new Matrix(Weights.Width, Weights.Height);
            for (int y = 0; y < Weights.Height; y++)
                for (int x = 0; x < Weights.Width; x++)
                {
                    deriv_activation[x] += this.Weights[x, y] * error[y];
                    deriv_weights[x, y] = this.Activation[x] * error[y];
                }
            this.deriv_biases += error;
            this.deriv_weights += deriv_weights;
            return deriv_activation;
        }

        public void AddDeriv()
        {
            float alpha = 0.1f; //this is the learningrate

            weights -= deriv_weights * alpha;
            biases -= deriv_biases * alpha;

            this.deriv_biases = default;
            this.deriv_weights = default;
        }
    }
}
