//import java.util.ArrayList;
//import java.util.List;

public class Neuron {
	
	private double activated_output;	
	private double[] weights;
	double bias;
	private double[] v; // previous weight change for momentum
	private double biasWeight;
	public Neuron(int numLinks){

		weights = new double[numLinks];
		v = new double[numLinks];
		// initialize weights [-0.3, 0.3]
		for(int i = 0; i<numLinks;i++){
			weights[i] = -0.3 + Math.random()*(0.6-0.3);
		}
		bias = -1;
	}
	
//	public Neuron(Neuron oldneuron, int numLinks ) {
//		this.sigmoid = oldneuron.sigmoid;
//		weights = new double[numLinks];
//		System.arraycopy( oldneuron.weights, 0, this.weights, 0, numLinks);
//		this.v = oldneuron.v;
//	}
//	
//	public static Neuron copy( Neuron oldneuron,int numLinks ) {
//		return new Neuron(oldneuron, numLinks);
//	}
//	
	public void updateWeights(double[] update){
		this.weights = update;
	}
	
//	public void updateBias(double change){
//		this.bias = bias - change;
//	}
	
	public double getWeight(int index){
		return this.weights[index];
	}
	
	public double activatedOutput(double[] inputs){
		
		// net output
		double net_output = 0.0;
	//	System.out.println("weight in neuron");
		for(int i = 0; i<inputs.length;i++){
			net_output += inputs[i]*weights[i];
		}
		
		net_output += bias*weights[inputs.length];
		// sigmoid as activation function
		this.activated_output = 1.0 / (1 + Math.exp(-1.0 * net_output));
		return this.activated_output;
	}
	
	// get partial derivative of the error with respect to actual output
	public double pdErrorWRTOutput(double target){
		return -(target-activated_output);
	}
	
	// derivative of weightedSum with respect to input
	public double pdOutputWRTNetout(){
		return activated_output * (1.0 - activated_output);
		
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
	
	
}

