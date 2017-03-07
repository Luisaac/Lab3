class Plate{
	double[][] matrix1;
	double[][] matrix2;
	double[][] inactivated;
	double[][] kernal;
	boolean[][] useAsMax;
	boolean[][] dropout1;
	boolean[][] dropout2;

	public Plate(int input_length, double[][] kernal, int kernal_length){
		matrix1 = new double[input_length][input_length];
		inactivated = new double[input_length][input_length];
		this.kernal = kernal;
		dropout1 = new boolean[input_length][input_length];

		useAsMax = new boolean[input_length][input_length];




	}

	public void output(int len){
//		int newLength = matrix1.length/len;
//		dropout2 = new boolean[newLength][newLength];
//
//		for(int i = 0; i < dropout1.length;i++){
//			for(int j = 0; j < dropout1.length;j++){
//				dropout1[i][j] = Math.random()<0.5? false :true; 
//			}
//		}
//		for(int i = 0; i < dropout2.length;i++){
//			for(int j = 0; j < dropout2.length;j++){
//				dropout2[i][j] = Math.random()<0.5? false :true; 
//			}
//		}

		matrix2= maxPooling(matrix1,len,false);
	}



	private double[][] maxPooling(double[][] matrix, int len, boolean overlap){
//		if(overlap){
//			int newLength = matrix.length-len+1;
//			double[][] newMatrix = new double[newLength][newLength];
//			for(int i = 0; i < newLength; i++){
//				for(int j = 0; j < newLength; j++){
//					newMatrix[i][j] = maxOfMatrix(matrix,i,j,len);
//				}
//			}
//			return newMatrix;
//		}
//		else{
		//if(matrix.length%len != 0) return null;
			//System.out.println("max index: ");
			int newLength = matrix.length/len;
			double[][]newMatrix = new double[newLength][newLength];
			for(int i = 0; i < matrix.length; i+=len){
				for(int j = 0; j < matrix.length; j+=len){
					newMatrix[i/len][j/len] = maxOfMatrix(matrix,i,j,len);
				}
			}
			
			return newMatrix;
		//}		
	}


	private double maxOfMatrix(double[][] matrix, int i, int j, int len){
		double max = Double.MIN_VALUE;
		int i_max = i;
		int j_max = j;
		
		for(int m = i; m< i+len;m++){
			for(int n = j; n<j+len;n++){
				if(matrix[m][n] > max){
					max = matrix[m][n];
					i_max = m;
					j_max = n;					
				}
			}
		}
		//TODO next image might overwrite values in the previous image
		useAsMax[i_max][j_max] = true;
		return max;
	}

}

