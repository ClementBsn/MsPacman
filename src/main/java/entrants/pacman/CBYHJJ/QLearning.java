package entrants.pacman.CBYHJJ;

import pacman.game.Constants;
import java.util.HashMap;

import static entrants.pacman.CBYHJJ.Tools.moveToIndex;

public class QLearning {

    private HashMap<Constants.MOVE,Integer> move_map;
    private double alpha = 0.0005; // learning rate : training = 0.0005 / test = 0
    private double gamma = 0.95; // discount factor
    private double to = 0.5; // exploration rate from 1 to 0

    public QLearning(double alpha, double gamma, double to){
        if (alpha >= 0 && alpha <= 1) {
            this.alpha = alpha;
        }

        if (gamma >= 0 && gamma <= 1) {
            this.gamma = gamma;
        }

        if (to >= 0 && to <= 1) {
            this.to = to;
        }

        int i = 0;
        move_map = new HashMap<Constants.MOVE, Integer>();
        for (Constants.MOVE move: Constants.MOVE.values()
             ) {
            move_map.put(move, i);
            i+=1;
        }
    }

    public double updateQValue(double[] qValues, int reward, boolean gameOver){

        if(gameOver)
        {
            return reward;
        }
        else
        {
            return reward + gamma * this.getBestQValue(qValues);
        }

    }

    public double getBestQValue(double[] qValues){
        int bestAction = -1;

        for (int i=0; i<qValues.length; i++){
            if (bestAction == -1 || qValues[bestAction] < qValues[i]){
                bestAction = i;
            }
        }

        return qValues[bestAction];
    }


    public Constants.MOVE getSoftMaxAction(double[] qValues){

        double[] softMax = new double[qValues.length];
        double softMaxSum = 0;



        for (int i=0; i<qValues.length; i++){
            softMax[i] = Math.exp(qValues[i]/to);
            softMaxSum += softMax[i];
        }

        DistributedRandomNumberGenerator drng = new DistributedRandomNumberGenerator();

        for (int i=0; i<qValues.length; i++){
            drng.addNumber(i, softMax[i]/softMaxSum); // Adds the numerical value 1 with a probability of 0.3 (30%)
        }

        int softMaxAction = drng.getDistributedRandomNumber(); // Generate an action according to the softmax distribution

        Constants.MOVE softMaxMove = null;
        for (Constants.MOVE move: Constants.MOVE.values()
        ) {
            int index = moveToIndex(move);
            if(index>-1) {
                if (index ==softMaxAction){
                    softMaxMove = move;
                    break;
                }
            }
        }
        return softMaxMove;
    }
}

