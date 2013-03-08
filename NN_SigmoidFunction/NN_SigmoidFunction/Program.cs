//  ---------------------------------------------------------------------------
//
//  @file       Program.cs
//  @brief      Program to train and test a neural network.
//
//  
//  @author     Harsha Nihanth N - http://www.antisphere.com
//  @date       02/25/2013
//
//  Compilation:
//  Find the Instructions in Readme File present inside this project.
//  ---------------------------------------------------------------------------


using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace NN_SigmoidFunction
{
    class Program
    {
        static void Main(string[] args)
        {
            string TrainfilePath="", TestFilePath ="";
            double learningRate = 0.0;
            int iterationCount = 0;
            DataTable table = new DataTable("data");
            try
            {
                 TrainfilePath = args[0];
                 TestFilePath = args[1];
                 learningRate = Convert.ToDouble(args[2]);
                 iterationCount = Convert.ToInt32(args[3]);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error in the input Please verify your input :\n" + ex.Message);
            }
            
            //Read Data from Input file
            table = ReadFiles(TrainfilePath);

            double[] weights = new double[(table.Columns.Count-1)];
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 0;
            }

            TrainData(table, weights, learningRate, iterationCount);

            DataTable testTable = new DataTable("TestData");
            testTable = ReadFiles(TestFilePath);

            // Calculate the Failed Count on the Train Data
            int trainFailures = TestData(table, weights);
            // Calculate the Failed Count on the Test Data
            int testFailures = TestData(testTable, weights);  
            //Total Records inside the Train Data
            int totalRecordsTrain = table.Rows.Count;
            //Total Records inside the Test Data
            int totalRecordsTest = testTable.Rows.Count;

            //Printing the Result
            Console.WriteLine("\n\nResult");
            Console.WriteLine("\n\nTrain Data:");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine("Total Records of Train: " + totalRecordsTrain);
            Console.WriteLine("Total Failure Records of Train : " + trainFailures);
            double accuracyTrain =(double)(totalRecordsTrain - trainFailures) * 100 / totalRecordsTrain;
            Console.WriteLine("Accuracy of Train :  " + accuracyTrain +"%");

            Console.WriteLine("\nTest Data :");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine("Total Records of Test : " + totalRecordsTest.ToString());
            Console.WriteLine("Total Failure Records of Test : " + testFailures);
            double accuracyTest =(double)(totalRecordsTest - testFailures) * 100 / totalRecordsTest;
            Console.Write("Accuracy of Test :" + accuracyTest + "%");

            Console.ReadLine();


        }

        /// <summary>
        /// Returns the total no of Failure cases for a give set of data.
        /// </summary>
        /// <param name="_table">Table containig the Data of either Test or Train</param>
        /// <param name="_weights">Weights</param>
        /// <returns>Failure Case Counts</returns>

        private static int TestData(DataTable _table, double[] _weights)
        {

            DataTable Table = _table;

            double summation = 0.0;
            double sigmoid = 0.0;
            int failCount =0;
            int result=0;
            double[] weights = _weights;
            foreach (DataRow row in Table.Rows)
            {
                result = 0;
                summation = 0;
                for (int i = 0; i < Table.Columns.Count-1; i++)
                {
                    double rowValue = Convert.ToDouble(row[i]);
                    summation+=(weights[i]*rowValue);   
                }
                //Calculation of Sigmoid
                sigmoid = 1 / (1 + Math.Exp(-(summation)));
                if (sigmoid>=0.5)
                {
                    result = 1;
                }

                int res = Convert.ToInt32(row["Result"]);
                if (result != Convert.ToInt32(row["Result"]))
                {
                    failCount++;
                }
            }
            return failCount;
        }
        /// <summary>
        /// This trains the Data wiht the train file, learning rate and number of Iterations mentioned
        /// </summary>
        /// <param name="_table">table storing the Train files data</param>
        /// <param name="_weights">Weights of the nodes</param>
        /// <param name="learnRate">learning rate Specified by the user</param>
        /// <param name="_iteration">No of Interations the Train data to be Iterated</param>
        private static void TrainData(DataTable _table, double[] _weights, double learnRate, int _iteration)
        {
            int rowCount=_iteration;
            double learningRate = learnRate;
            double summation = 0.0;
            DataTable table = _table;
            double[] weights = _weights;
            
            int nTimes = rowCount / table.Rows.Count;
            int balance = rowCount % table.Rows.Count;

            double rowValue = 0.0;
            while (nTimes!=0)
            {
                for (int i = 0; i < table.Rows.Count; i++)
                {
                    summation = 0;
                    for (int j = 0; j < table.Columns.Count - 1; j++)
                    {
                        rowValue = Convert.ToDouble(table.Rows[i][j]);
                        summation += (weights[j] * rowValue);
                    }
                    //Sigmoid Calculation
                    double Sigmoid = 1 / (1 + Math.Exp(-(summation)));
                    //string bcd = table.Rows[i][table.Columns["Result"]].ToString();
                    //Error Calculation
                    double Err = Convert.ToInt32(table.Rows[i][table.Columns["Result"]]) - Sigmoid;
                    //Sigmoid Prime Calculations
                    double sigMoidPrime = Math.Exp(summation) / Math.Pow((1 + Math.Exp(summation)), 2);
                    //Updating Weights
                    for (int k = 0; k < table.Columns.Count - 1; k++)
                    {
                        weights[k] = weights[k] + learningRate * Err * sigMoidPrime * Convert.ToDouble(table.Rows[i][k]);
                    }
                }
                nTimes--;

            }
            for (int i = 0; i < balance; i++)
            {
                summation = 0;
                for (int j = 0; j < table.Columns.Count - 1; j++)
                {
                     rowValue = Convert.ToDouble(table.Rows[i][j]);
                    summation += (weights[j] * rowValue);
                }
                //Sigmoid Calculation
                double Sigmoid = 1 / (1 + Math.Exp(-(summation)));
                //Error Calculation
                double Err = Convert.ToInt32(table.Rows[i][table.Columns["Result"]]) - Sigmoid;
                //Sigmoid Prime Calculation
                double sigMoidPrime = Math.Exp(summation) / Math.Pow((1 + Math.Exp(summation)), 2);
                //Weights Updation
                for (int k = 0; k < table.Columns.Count - 1; k++)
                {
                    weights[k] = weights[k] + learningRate * Err * sigMoidPrime * Convert.ToDouble(table.Rows[i][k]);
                }
            }
            
        }

        /// <summary>
        /// Reads Input file given and stores it inside the DataTable
        /// </summary>
        /// <param name="InputfilePath"></param>
        /// <returns>DataTable containing file Data</returns>
        private static DataTable ReadFiles(string filePath)
        {
            DataTable table = new DataTable("Data");
            TextReader reader = new StreamReader(filePath);
            bool colAdded = false;
            try
            {
                while (reader.Peek() != -1)
                {
                    string[] tokens = Regex.Split(reader.ReadLine(), "[\t\r\n]");

                    //Adding the row X0
                    Array.Resize(ref tokens,tokens.Length+1);
                    tokens[tokens.Length-1] = "1";

                    //Column Addition inside the DataTbale
                    if (!colAdded)
                    {
                        foreach (string token in tokens)
                        {
                            if (token == "1")
                            {
                                table.Columns.Add("X0");
                                break;
                            }
                            else if (token == "")
                            {
                                table.Columns.Add("Result");
                            }
                            else
                            {
                                table.Columns.Add(token);
                            }
                            
                        }
                        colAdded = true;
                        
                    }
                        
                    //Adding Data inside the  Rows of DataTable
                    else
                    {
                        DataRow row = table.NewRow();
                        for (int i = 0; i < table.Columns.Count; i++)
                        {
                            row[i] = tokens[i];
                        }
                        table.Rows.Add(row);
                        
                    }
                }
            }
            catch (IndexOutOfRangeException)
            {
               //Exception Handler
            }
            finally
            {
                //Close the reader
                if (reader != null)
                {
                    reader.Close();
                }
            }
            //Set the X0 columns as the First column of the Data Table
            table.Columns["X0"].SetOrdinal(0);
            return table;
        }
    }
}
