class Plate{
	double[][] matrix1;
	double[][] matrix2;
	double[][] inactivated;
	double[][][] kernal;
	boolean[][] useAsMax;
	boolean[][] dropout1;
	boolean[][] dropout2;
	int input_length;

	public Plate(int input_length, double[][][] kernal, int kernal_length){
		matrix1 = new double[input_length][input_length];
		matrix2 = new double[input_length/2][input_length/2];
		inactivated = new double[input_length][input_length];
		this.kernal = kernal;
		dropout1 = new boolean[input_length][input_length];

		useAsMax = new boolean[input_length][input_length];
		this.input_length = input_length;
	}

	public void output(int len){
		//double[][]newMatrix = new double[newLength][newLength];
		//TODO
//		if(this.input_length == 5){
//			len = 1;
//			//System.out.println(matrix1.length);
//		}
		
		
		for(int i = 0; i < matrix1.length; i+=len){
			for(int j = 0; j < matrix1.length; j+=len){
				matrix2[i/len][j/len] = maxOfMatrix(matrix1,i,j,len);
			}
		}
	}



	private double maxOfMatrix(double[][] matrix, int i, int j, int len){
		double max = matrix[i][j];
		int i_max = i;
		int j_max = j;
		//TODO
//		if(this.input_length == 5){
//			len = 1;
//			//System.out.println(matrix1.length);
//		}
		for(int m = i; m < i+len;m++){
			for(int n = j; n<j+len;n++){
				if(matrix[m][n] > max){
					max = matrix[m][n];
					i_max = m;
					j_max = n;					
				}
			}
		}
		
		useAsMax[i_max][j_max] = true;
		return max;
	}
	
	

}

