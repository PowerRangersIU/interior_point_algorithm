import java.util.Arrays;

public class InteriorPoint {

    public static void main(String[] args) {
        double[] C = {1, 1}; // минимизация x + y
        double[][] A = {
                {1, 0}, // x >= 1
                {0, 1}  // y >= 1
        };
        double[] b = {0.5, 0.5}; // Ограничение, которое невозможно выполнить
        double[] xInitial = {0, 0};


        double eps = 1e-6;
        double alpha1 = 0.5;
        double alpha2 = 0.9;

        // Solve the problem
        Result result1 = interiorPointMethod(C, A, eps, alpha1, xInitial);
        Result result2 = interiorPointMethod(C, A, eps, alpha2, xInitial);

        // Print the result
        System.out.println("Solution with alpha = 0,5:");
        System.out.println(result1);
        System.out.println();
        System.out.println("Solution with alpha = 0,9:");
        System.out.println(result2);
    }

    public static Result interiorPointMethod(double[] C, double[][] A, double eps, double alpha, double[] xInitial) {
        int n = C.length;
        double[][] I = identityMatrix(n);
        double[] x = xInitial.clone();
        int iteration = 0;
        int maxIteration = 35;

        while (true) {
            double[] xPrev = x.clone();
            double[][] D = new double[n][n];
            for (int i = 0; i < n; i++) {
                D[i][i] = x[i];
            }

            double[][] A_hat = multiplyMatrices(A, D);
            double[] c_hat = multiplyMatrixVector(D, C);

            double[][] P;
            try {
                double[][] F = multiplyMatrices(A_hat, transpose(A_hat));

                if (determinant(F) == 0) {
                    return new Result("The matrix F is singular and cannot be inverted.");
                }

                double[][] F_inv = invertMatrix(F);
                double[][] H = multiplyMatrices(transpose(A_hat), multiplyMatrices(F_inv, A_hat));
                P = subtractMatrices(I, H);
            } catch (Exception e) {
                return new Result("The method is not applicable!");
            }

            double[] c_p = multiplyMatrixVector(P, c_hat);

            boolean feasible = true;
            for (double cp : c_p) {
                if (cp < 0) {
                    feasible = false;
                    break;
                }
            }
            if (feasible) {
                return new Result("The problem does not have a solution!");
            }

            double nu = 0;
            for (double cp : c_p) {
                nu = Math.max(nu, -cp);
            }

            double[] x_hat = new double[n];
            for (int i = 0; i < n; i++) {
                x_hat[i] = 1 + (alpha / nu) * c_p[i];
            }
            x = multiplyMatrixVector(D, x_hat);

            iteration++;

            if (iteration > maxIteration) {
                break;
            }
            if (euclideanNorm(subtractVectors(x, xPrev)) < eps) {
                break;
            }
        }

        double optimalValue = 0;
        for (int i = 0; i < n; i++) {
            optimalValue += C[i] * x[i];
        }

        return new Result(x, optimalValue, iteration);
    }

    private static double[][] identityMatrix(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) {
            I[i][i] = 1;
        }
        return I;
    }

    private static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposed = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    private static double[][] multiplyMatrices(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = b[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < a[0].length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    private static double[] multiplyMatrixVector(double[][] matrix, double[] vector) {
        int rows = matrix.length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    private static double[][] subtractMatrices(double[][] a, double[][] b) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    private static double[] subtractVectors(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] - b[i];
        }
        return result;
    }

    private static double euclideanNorm(double[] vector) {
        double sum = 0;
        for (double v : vector) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    private static double determinant(double[][] matrix) {
        int n = matrix.length;
        if (n == 1) {
            return matrix[0][0];
        }

        if (n == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }

        double det = 0;
        for (int col = 0; col < n; col++) {
            double[][] minor = new double[n - 1][n - 1];

            for (int i = 1; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (j < col) {
                        minor[i - 1][j] = matrix[i][j];
                    } else if (j > col) {
                        minor[i - 1][j - 1] = matrix[i][j];
                    }
                }
            }
            det += Math.pow(-1, col) * matrix[0][col] * determinant(minor);
        }
        return det;
    }

    private static double[][] invertMatrix(double[][] matrix) {
        int n = matrix.length;
        double[][] augmented = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, augmented[i], 0, n);
            augmented[i][i + n] = 1;
        }

        for (int i = 0; i < n; i++) {
            double pivot = augmented[i][i];
            if (pivot == 0) throw new ArithmeticException("Matrix is singular and cannot be inverted.");

            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }

            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(augmented[i], n, inverse[i], 0, n);
        }
        return inverse;
    }

    public static class Result {
        public String message;
        public double[] solution;
        public double optimalValue;
        public int iterations;

        public Result(String message) {
            this.message = message;
        }

        public Result(double[] solution, double optimalValue, int iterations) {
            this.solution = solution;
            this.optimalValue = optimalValue;
            this.iterations = iterations;
        }

        @Override
        public String toString() {
            if (message != null) return message;
            StringBuilder solutionStr = new StringBuilder("Solution: ");
            for (double s : solution) {
                solutionStr.append(s).append(" ");
            }
            return solutionStr + "\nOptimal value: " + optimalValue + "\nIterations: " + iterations;
        }
    }

}
