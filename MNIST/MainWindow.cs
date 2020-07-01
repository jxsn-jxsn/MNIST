using MNIST.Properties;
using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml.Serialization;

namespace MNIST
{
    public partial class MainWindow : Form
    {
        private Dictionary<Vector, Vector> trainingData;
        private List<float> acc;
        private Network network;

        private const int IMG_SIZE = 28;

        public MainWindow()
        {
            InitializeComponent();
            DoubleBuffered = true;
            trainingData = new Dictionary<Vector, Vector>();
            acc = new List<float>();
            network = new Network(IMG_SIZE * IMG_SIZE, 16, 16, 10);
            network.OnTrack += Network_OnTrack;
            LoadTrainingData();
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            string path = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
            Serialize(acc, path + @"\chart.xml");
            Serialize(network, path + @"\network.xml");
            base.OnFormClosed(e);
        }

        private void Network_OnTrack(object sender, float e)
        {
            acc.Add(e * 100);
            this.Refresh();
        }

        private void LoadTrainingData()
        {
            string[] data = Resources.train.Split("\n");
            Parallel.ForEach(data, (string line) =>
            {
                string[] digit = line.Split(",");
                if (digit.Length <= 1)
                    return;
                Vector input = new Vector(digit.Length - 1);
                for (int i = 1; i < digit.Length; i++)
                    input[i - 1] = int.Parse(digit[i]) / (float)255.0f;
                Vector output = new Vector(10);
                int label = int.Parse(digit[0]);
                output[label] = 1;
                lock (trainingData)
                    trainingData.Add(input, output);
            });
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            RenderChart(e.Graphics);
            RenderNetwork(e.Graphics);
        }

        private void RenderChart(Graphics g)
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            float sizeX = 400, sizeY = 200;
            g.TranslateTransform(30, 30);
            g.DrawLine(Pens.Black, 0, 0, 0, sizeY);
            g.DrawLine(Pens.Black, 0, sizeY, sizeX, sizeY);

            int max = acc.Count;
            int prevIndex = 0, prevI = 0;

            if (acc.Count >= 1)
                for (int i = 0; i < sizeX; i++)
                {
                    int index = (int)(i / (float)sizeX * acc.Count);
                    if (index <= 0)
                        continue;
                    g.DrawLine(Pens.Green, prevI, sizeY - acc[prevIndex] * 2, i, sizeY - acc[index] * 2);
                    prevIndex = index;
                    prevI = i;
                }

            for (int i = 0; i <= 100; i+= 25)
                g.DrawString(i.ToString(), new Font("Arial", 8), Brushes.Black, -25, (sizeY - i) * 2 - 200);

            for (int i = 1; i <= 4; i++)
                g.DrawString((max / 4 * i).ToString(), new Font("Arial", 8), Brushes.Black,  sizeX / 4 * i, sizeY + 25);
        }

        private void RenderNetwork(Graphics g)
        {
            g.ResetTransform();
            for (int k = 0; k < network.Layer.Length; k++)
            {
                float scale = 2.5f;
                int margin = 10;
                float x = 450;
                float y = 30 + k * 100;
                for (int i = 0; i < network[k].Connections; i++)
                {
                    Bitmap bitmap = GetBitmap(network[k], i);
                    if (i > 0)
                        x += bitmap.Width * scale + margin;
                    if (x + bitmap.Width * scale + margin > Width)
                    {
                        x = 450;
                        y += bitmap.Height * scale + margin;
                    }
                    g.TranslateTransform(30 + x, y);
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;
                    g.ScaleTransform(scale, scale);
                    g.DrawImage(bitmap, 0, 0);
                    g.ResetTransform();
                }
            }
        }

        private Bitmap GetBitmap(Layer layer, int index)
        {
            MNIST.Utils.Matrix weights = layer.Weights;
            int size = (int)Math.Sqrt(weights.Width);
            Bitmap bitmap = new Bitmap(size, size);
            for(int x = 0; x < size; x++)
                for(int y = 0; y < size; y++)
                {
                    float value = weights[x + y * size, index] * 100;
                    if (value > 255)
                        value = 255;
                    Color color = Color.Green;
                    if (value < 0)
                    {
                        if(value < -255)
                            value = -255;
                        color = Color.Red;
                    }
                    bitmap.SetPixel(x, y, Color.FromArgb((int)Math.Abs(value), color));
                }
            return new Bitmap(bitmap, size, size);
        }

        private void btnTrain_Click(object sender, EventArgs e)
        {
            //trainingData = network.GetBatch(trainingData, 0, 1000);
            network.Train(trainingData, Network.Track.ACC, Network.Optimizer.MiniBatch, 2, 12);
        }

        private void Serialize<T>(T t, string path)
        {
            using (FileStream stream = File.Open(path, FileMode.Create))
            {
                new XmlSerializer(typeof(T)).Serialize(stream, t);
            }
        }

        private T Deserialize<T>(string path)
        {
            using (FileStream stream = File.Open(path, FileMode.Open))
                return (T)new XmlSerializer(typeof(T)).Deserialize(stream);
        }

        private void btnOpen_Click(object sender, EventArgs e)
        {
            using(OpenFileDialog dialog = new OpenFileDialog())
            {
                if (dialog.ShowDialog() == DialogResult.Cancel)
                    return;
                try
                {
                    acc = Deserialize<List<float>>(dialog.FileName);
                }
                catch (Exception)
                {

                }

                try
                {
                    Network network = Deserialize<Network>(dialog.FileName);
                    this.network = network;
                    network.OnTrack += Network_OnTrack;
                }
                catch (Exception)
                {

                }
                this.Refresh();
            }
        }
    }
}
