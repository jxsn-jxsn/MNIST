using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    [Serializable]
    public class Network
    {
        private Layer[] layer;
        public event EventHandler<float> OnTrack;

        public Layer[] Layer { get => layer; set => layer = value; }

        public Layer this[int index]
        {
            get
            {
                return layer[index];
            }
        }

        public Network(params int[] layer)
        {
            this.layer = new Layer[layer.Length - 1]; //last layer is just output
            for (int i = 0; i < layer.Length - 1; i++)
                this.layer[i] = new Layer(layer[i], layer[i + 1]);
        }

        public Network()
        {

        }

        public Vector GetOutput(Vector input)
        {
            for (int i = 0; i < layer.Length; i++)
                input = this[i].GetOutput(input);
            return input;
        }

        public int GetOutputIndex(Vector input)
        {
            Vector output = GetOutput(input);
            int index = -1;
            for (int i = 0; i < output.Dimensions; i++)
                if (index == -1 || output[i] > output[index])
                    index = i;
            return index;
        }

        public float BackpropBatch(Dictionary<Vector, Vector> batch, Track track)
        {
            float sum = 0;
            object lockObject = new object();

            Parallel.ForEach(batch.Keys, (Vector key) =>
            {
                var (acc, loss) = Backprop(key, batch[key]);
                lock (lockObject) 
                {
                    sum += track switch
                    {
                        Track.ACC => acc,
                        Track.LOSS => loss,
                        _ => 0
                    };
                }
            }
            );

            AddDeriv();

            return sum / (float)batch.Count;
        }

        public (float, float) Backprop(Vector input, Vector output)
        {
            Vector loss = GetOutput(input) - output;
            Vector error = loss * 2;

            for (int i = Layer.Length - 1; i >= 0; i--)
                error = this[i].Backprop(error);

            return (output[GetOutputIndex(input)], (loss * loss).Sum());
        }

        public Dictionary<Vector, Vector> GetBatch(Dictionary<Vector, Vector> trainingData, int index, int size)
        {
            Dictionary<Vector, Vector> batch = new Dictionary<Vector, Vector>();
            for(int i = 0; i < size; i++)
            {
                if (index + i >= trainingData.Count)
                    break;
                KeyValuePair<Vector, Vector> pair = trainingData.ElementAt(index + i);
                batch.Add(pair.Key, pair.Value);
            }
            return batch;
        }

        public void AddDeriv()
        {
            for(int i = 0; i < layer.Length; i++)//parallel?
                this[i].AddDeriv();
        }

        public void Train(Dictionary<Vector, Vector> trainingData, Track track, Optimizer optimizer, int epochs, int batch_size = 64)
        {
            for(int i = 0; i < epochs; i++)
            {
                switch (optimizer)
                {
                    case Optimizer.GradientDescent:
                        float track_value = BackpropBatch(trainingData, track);
                        OnTrack?.Invoke(this, track_value);
                        break;
                    case Optimizer.MiniBatch:
                        int runs = 0;
                        track_value = 0;
                        for(int index = 0; index < trainingData.Count; index+= batch_size)
                        {
                            Dictionary<Vector, Vector> batch = GetBatch(trainingData, index, batch_size);
                            track_value += BackpropBatch(batch, track);
                            runs++;
                            OnTrack?.Invoke(this, (track_value / (float)runs));
                        }
                        break;
                }
            }
        }

        public enum Track : int
        {
            ACC = 0,
            LOSS = 1
        }

        public enum Optimizer : int
        {
            GradientDescent = 0,
            MiniBatch = 1
        }

    }
}
