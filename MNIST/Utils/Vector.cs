using System;
using System.Collections.Generic;
using System.Text;

namespace MNIST.Utils
{
    public struct Vector
    {
        private float[] values;

        public float[] Values { get => values; set => values = value; }

        public Vector(params float[] values)
        {
            this.values = values;
        }

        public Vector(int dimensions) => values = new float[dimensions];

        public int Dimensions => values.Length;

        public float this[int index]
        {
            get
            {
                return values[index];
            }
            set
            {
                values[index] = value;
            }
        }

        public static Vector operator +(Vector v1, float v2)
        {
            for (int i = 0; i < v1.Dimensions; i++)
                v1[i] += v2;
            return v1;
        }

        public static Vector operator *(Vector v1, float v2)
        {
            for (int i = 0; i < v1.Dimensions; i++)
                v1[i] *= v2;
            return v1;
        }

        public static Vector operator /(Vector v1, float v2)
        {
            for (int i = 0; i < v1.Dimensions; i++)
                v1[i] /= v2;
            return v1;
        }

        public static Vector operator *(Vector v1, Vector v2)
        {
            for (int i = 0; i < v1.Dimensions; i++)
                v1[i] *= v2[i];
            return v1;
        }

        public static Vector operator /(Vector v1, Vector v2)
        {
            for (int i = 0; i < v1.Dimensions; i++)
                v1[i] /= v2[i];
            return v1;
        }

        public static Vector operator +(Vector v1, Vector v2)
        {
            if (v1.Equals(default(Vector)))
                v1 = new Vector(v2.Dimensions);

            for (int i = 0; i < v1.Dimensions; i++)
                v1[i] += v2[i];
            return v1;
        }

        public static Vector operator -(Vector v1, Vector v2)
        {
            if (v1.Equals(default(Vector)))
                v1 = new Vector(v2.Dimensions);

            for (int i = 0; i < v1.Dimensions; i++)
                v1[i] -= v2[i];
            return v1;
        }

        public Vector ForEach(Func<float, float> func)
        {
            for (int i = 0; i < this.Dimensions; i++)
                this[i] = func(this[i]);
            return this;
        }

        public float Sum()
        {
            float sum = 0;
            for (int i = 0; i < Dimensions; i++)
                sum += this[i];
            return sum;
        }
    }
}
