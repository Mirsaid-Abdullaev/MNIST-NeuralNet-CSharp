using System;

namespace NeuralNetwork
{
    internal static class MathUtil
    {
        public static double Sigmoid(double x)
        {
            if (x > 40)
            {
                return 1;
            }
            else if (x < -40)
            {
                return 0;
            }
            return 1 / (1 + Math.Exp(-1 * x));
        }
        public static double Sigmoid_Deriv(double x)
        {
            if (x > 40 || x < -40)
            {
                return 0;
            }
            double y = Sigmoid(x); //y = sigmoid(x)
            return y * (1 - y); //d(sigmoidx)/dx = y(1-y)
        }
        public static double ReLU(double x)
        {
            return Math.Max(x, 0);
        }
        public static double ReLU_Deriv(double x)
        {
            return Convert.ToDouble(x > 0);
        }
        public static double Tanh(double x)
        {
            if (x > 40)
            {
                return 1;
            }
            else if (x < -40)
            {
                return -1;
            }
            return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        }
        public static double Tanh_Deriv(double x)
        {
            if (x > 40 || x < -40)
            {
                return 0;
            }
            return 1 / Square((Math.Exp(x) + Math.Exp(-x)) / 2);
        }
        public static double Square(double x)
        {
            return x * x;
        }
        public static double LeakyRELU(double x)
        {
            if (x < 0)
            {
                return x * 0.24;
            }
            else
            {
                return x;
            }
        }
        public static double LeakyRELU_Deriv(double x)
        {
            if (x < 0)
            {
                return 0.24;
            }
            else
            {
                return 1;
            }
        }
        public static double[] CreateOneHot(byte Label, int Length)
        {
            if (Label >= Length)
            {
                throw new Exception($"Error: label number \"{Label}\" is not within 0-{Length}, cannot create a {Length}-length One Hot formatted array from this digit.");
            }
            double[] Result = null;
            if (Length == 10) //a faster optimised function for length 10 - only done for 10 as currently only using 10. Feel free to add another case to this.
            {
                switch (Label) 
                {
                    case 0:
                        Result = new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                        break;
                    case 1:
                        Result = new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
                        break;
                    case 2:
                        Result = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
                        break;
                    case 3:
                        Result = new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
                        break;
                    case 4:
                        Result = new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
                        break;
                    case 5:
                        Result = new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 };
                        break;
                    case 6:
                        Result = new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 };
                        break;
                    case 7:
                        Result = new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 };
                        break;
                    case 8:
                        Result = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 };
                        break;
                    case 9:
                        Result = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
                        break;
                }
            }
            else
            {
                Result = new double[Length];
                for (int i = 0; i < Length; i++)
                {
                    Result[i] = (i == Label) ? 1 : 0; // item = 1 if the index is same as label, 0 otherwise
                }
            }

            return Result;
        }
    }
}
