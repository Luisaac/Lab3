import java.util.Random;

//import java.util.ArrayList;
//import java.util.List;

public class Neuron {
	public static Random rand = new Random(8000);
	private double activated_output;	
	private double[] weights;
	double bias;
	private double[] v; 
	private double[] mpar;
	private double[] vpar;
	private int[] tpar;
	
	// previous weight change for momentum
	private int actFunc; // 0-leaky, 1-soft, 2-sigmoid
	public Neuron(int numLinks, int actFunc){
		this.actFunc = actFunc;
		weights = new double[numLinks];
		v = new double[numLinks];
		mpar = new double[numLinks];
		vpar = new double[numLinks];
		tpar = new int[numLinks];
		
		// initialize weights [-0.3, 0.3]
		for(int i = 0; i<numLinks;i++){
			weights[i] = Lab3.getRandomWeight(numLinks, 0);
			//TODO
			//weights[i] = 1.0;
		}
		bias = -1;
	}

	public void updateWeights(double[] update){
		this.weights = update;
	}
	
	
	public double getWeight(int index){
		return this.weights[index];
	}
	
	public double activatedOutput(double[] inputs){
		
		// net output
		double net_output = 0.0;
		for(int i = 0; i<inputs.length;i++){
			net_output += inputs[i]*weights[i];
		}
		
		net_output += bias*weights[inputs.length];
		// sigmoid as activation function
		if(actFunc == 2){
			this.activated_output = 1.0 / (1 + Math.exp(-1.0 * net_output));
		}else if(actFunc == 0){
			
			this.activated_output = (net_output >0)?net_output:net_output*0.01;
//			if(weights.length > 350)
//				System.out.println("activated: "+this.activated_output);
		}else
			this.activated_output = net_output;
		return this.activated_output;
	}
	
	// get partial derivative of the error with respect to actual output
	public double pdErrorWRTOutput(double target){
		return -(target-activated_output);
	}
	
	
	// derivative of weightedSum with respect to input
	public double pdOutputWRTNetout(){
		if(actFunc == 2){
			return activated_output * (1.0 - activated_output);
		}else if(actFunc == 0){
			return activated_output>0?1:0.01;
		}else{
			return 0;
		}
	}
	
	public double pdErrorWRTNetout(double target){
		return pdErrorWRTOutput(target)*pdOutputWRTNetout();
	}
	
	public double computeError(double target){
		return 1/2*Math.pow((target-this.activated_output), 2);
	}
	
	public double getV(int i){
		return v[i];
	}
	
	public void setV(double v, int i){
		this.v[i] = v;
	}
	
	public double getParM(int i) {
		return mpar[i];
	}
	
	public void setParM(double mpar , int i){
		this.mpar[i] = mpar;
	}

	public double getParV(int i){
		return vpar[i];
	}
	
	
	public void setParV(double vpar, int i){
		this.vpar[i] = vpar;
	}
	
	public int getParT( int i ) {
		return this.tpar[i];
	}
	
	public void setParT(int tpar, int i) {
		this.tpar[i] = tpar + 1;
	}
	
	
	
	
}

