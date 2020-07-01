using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

namespace MNIST.Utils
{
    public struct Matrix
    {
        private float[][] values;

        public int Width => Values.Length;
        public int Height => Values.First().Length;

        public float[][] Values { get => values; set => values = value; }

        public Matrix(int width, int height)
        {
            this.values = new float[width][];

            for (int i = 0; i < width; i++)
                this.Values[i] = new float[height];
        }

        public float this[int x, int y]
        {
            get
            {
                return Values[x][y];
            }
            set
            {
                Values[x][y] = value;
            }
        }

        public static Vector operator *(Matrix m, Vector v)
        {
            Vector var1 = new Vector(m.Height);
            for (int x = 0; x < m.Width; x++)
                for (int y = 0; y < m.Height; y++)
                    var1[y] += v[x] * m[x, y];
            return var1;
        }

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            if (m1.Equals(default(Matrix)))
                m1 = new Matrix(m2.Width, m2.Height);

            for (int x = 0; x < m1.Width; x++)
                for (int y = 0; y < m1.Height; y++)
                    m1[x, y] += m2[x, y];
            return m1;
        }

        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            if (m1.Equals(default(Matrix)))
                m1 = new Matrix(m2.Width, m2.Height);
            for (int x = 0; x < m1.Width; x++)
                for (int y = 0; y < m1.Height; y++)
                    m1[x, y] -= m2[x, y];
            return m1;
        }

        public static Matrix operator *(Matrix m1, float m2)
        {
            for (int x = 0; x < m1.Width; x++)
                for (int y = 0; y < m1.Height; y++)
                    m1[x, y] *= m2;
            return m1;
        }


        public Matrix ForEach(Func<float, float> func)
        {
            for (int x = 0; x < Width; x++)
                for (int y = 0; y < Height; y++)
                    this[x, y] = func(this[x, y]);
            return this;
        }
    }
}
