using System;
using System.Collections.Generic;
using System.Text;

namespace MNIST.Utils
{
    public class MathUtils
    {

        public static float Sigmoid(float value)
        {
            return 1f / (1f + (float)Math.Pow(Math.E, -value));
        }

        public static float DerivSigmoid(float value)
        {
            return Sigmoid(value) * (1 - Sigmoid(value));
        }
    }
}
