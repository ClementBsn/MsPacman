package entrants.pacman.CBYHJJ;

import pacman.game.Constants;
import pacman.game.Game;


public class Tools {

    public static int getNearestJunction(Game game, Constants.MOVE c, double longest_path) {
        double minDistance = longest_path;
        int nearestJunctionIndex=-1;
        for (int i:game.getJunctionIndices()){
            if(i > -1 && game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), i, c)<minDistance)
            {
                nearestJunctionIndex=i;
                minDistance = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), i, c);
            }

        }

        return nearestJunctionIndex;
    }


    public static int getNearestGhostIndex(Game game, Constants.MOVE c, double longest_path) {
        double minDistance= longest_path;
        int nearestGhostIndex=-1;
        int i;
        for (Constants.GHOST type: Constants.GHOST.values()){
            i=game.getGhostCurrentNodeIndex(type);
            if(i > -1 && !game.isGhostEdible(type) && game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), i, c)<minDistance)
            {
                nearestGhostIndex=i;
                minDistance = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), i, c);
            }
        }

        return nearestGhostIndex;
    }


    public static double getLongestPath(Game game){
        double longestPath = -1;
        int[] distances = game.getCurrentMaze().shortestPathDistances;

        for(int i=0; i<distances.length; i++){
            if(longestPath < distances[i]){
                longestPath = distances[i];
            }
        }

        return longestPath;
    }


    public static boolean reversed(Constants.MOVE myMove, Constants.MOVE nextMove){
        if(moveToIndex(myMove) == 1 && moveToIndex(nextMove) == 2){
            return false;
        }
        if(moveToIndex(myMove) == 2 && moveToIndex(nextMove) == 1){
            return false;
        }
        if(Math.abs(moveToIndex(myMove) - moveToIndex(nextMove)) == 1){
            return true;
        }
        return false;
    }

    public static int processReward(Game game, Constants.MOVE myMove, Constants.MOVE nextMove){

        int reward = -5;

        if (reversed(myMove, nextMove)){
            reward -= 6;
        }

        if (game.getNumberOfActivePills()==0){ // if game ended and she won +50
            reward += 50;
        }

        if (isWithGhost(game) || game.wasPacManEaten()){
            reward -= 350;
        }
        // if collided with scared ghost +20
        int nEatenGhosts = 0;
        for(Constants.GHOST ghost : Constants.GHOST.values()){
            if(game.wasGhostEaten(ghost)){
                nEatenGhosts++;
            }
        }
        reward += nEatenGhosts*20;

        // if ate pill +12
        if (game.wasPillEaten()){
            reward += 12;
        }

        // if ate powerpill +3
        if (game.wasPowerPillEaten()){
            reward += 3;
        }

        return reward;
    }


    public static int moveToIndex(Constants.MOVE move){
        if(move == Constants.MOVE.UP){
            return 0;
        }
        if(move == Constants.MOVE.DOWN){
            return 1;
        }
        if(move == Constants.MOVE.LEFT){
            return 2;
        }
        if(move == Constants.MOVE.RIGHT){
            return 3;
        }

        return -1;
    }

    public static Constants.MOVE indexToMove(int index){
        if(index == 0){
            return Constants.MOVE.UP;
        }
        if(index == 1){
            return Constants.MOVE.DOWN;
        }
        if(index == 2){
            return Constants.MOVE.LEFT;
        }
        if(index == 3){
            return Constants.MOVE.RIGHT;
        }

        return Constants.MOVE.NEUTRAL;
    }

    public static boolean isWithGhost(Game game){

        int pacmanPos = game.getPacmanCurrentNodeIndex();

        for(Constants.GHOST ghost : Constants.GHOST.values()){
            int position = game.getGhostCurrentNodeIndex(ghost);
            if (position == pacmanPos){
                return true;
            }
        }

        return false;
    }

}
