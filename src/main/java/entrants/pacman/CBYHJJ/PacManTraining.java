package entrants.pacman.CBYHJJ;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import pacman.controllers.PacmanController;
import pacman.game.Constants.GHOST;
import pacman.game.Constants.MOVE;
import pacman.game.Constants;
import pacman.game.Game;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import static entrants.pacman.CBYHJJ.Tools.*;


public class PacManTraining extends PacmanController {
    private MOVE myMove = MOVE.NEUTRAL;
	private QLearning qLearning = null;
	private double longest_path = -1;
	private MultiLayerNetwork model = null;
	private INDArray qSaving = null;
	private double count = 0;
	private int nbMoves = 0;


	public PacManTraining(int i)
    {
        super();
        count = i;

    }
    public MOVE getMove(Game game, long timeDue) {

        /* Initialize the QLearning with different values of exploration_rate
        * to allow the PacMan to explore and find other rules in function of the games played
        */
        if(qLearning == null){
            double exploration_rate;

            if(count < 10){
                exploration_rate = 1;
            }
            else{
                if(count < 24) {
                    exploration_rate = 0.8;
                }
                else{
                    if (count >= 4500) {
                        exploration_rate = 0;
                    } else {
                        exploration_rate = 20 / count;
                    }
                }
            }

            qLearning = new QLearning(0.95, 0.0005, exploration_rate);
            longest_path = getLongestPath(game);
        }

        // Load the Neural Network
        if(model == null){
            model = loadNN(false);
        }

        // Data is used to save the Qvalues, Reward and other in data.txt
        String data = "########## MOVE "+nbMoves+" ##########\n\n-----------------------------------------------------\n\n";

        // 1 - calculer les paramètres
        double pillsEatenInput = this.PillsEatenInput(game);
        double powerPillInput = this.powerPillInput(game);
        double[] pillInput = this.pillInput(game);
        double[] ghostInput = this.ghostInput(game);
        double[] ghostAfraidInput = this.ghostAfraidInput(game);
        double[] entrapementInput = this.entrapmentInput(game);
        int[] isGoingTowards = this.isGoingToward();

        // 2 - calculer les QValeurs pour chaque direction en utilisant le réseau de neurones
        double[] qValues = new double[4];
        for (MOVE move: MOVE.values()){
            int index = moveToIndex(move);
            if(index > -1) {
                INDArray array = initializeINDarray(pillsEatenInput, powerPillInput, pillInput[index], ghostInput[index],
                        ghostAfraidInput[index], entrapementInput[index], isGoingTowards[index]);

                qValues[index] = model.output(array).getDouble(0,0);

                data += index+" | "+pillsEatenInput+" | "+powerPillInput+" | "+pillInput[index]+" | "+ghostInput[index]
                        +" | "+ghostAfraidInput[index]+" | "+entrapementInput[index]+" | "+isGoingTowards[index]+" | "
                        +qValues[index]+"\n";
            }
        }

        data += "\n------------------------------------------------------\n";

        // 3 - sélectionner l'action avec la meilleure QValeur calculée en 2 avec un softmax pour inciter à l'exploration
        MOVE nextMove = qLearning.getSoftMaxAction(qValues);

        // 4 - calculer la nouvelle QValeur de l'état précédent en se basant sur l'état courant
        if(qSaving!=null) {
            int reward = processReward(game, myMove, nextMove);
            boolean gameOver = (game.getActivePillsIndices().length == 0 || game.wasPacManEaten());
            double newQValue = qLearning.updateQValue(qValues, reward, gameOver);

            data += nextMove+" "+reward+" "+newQValue+"\n";


            // 5 - mettre à jour le réseau en fonction de la QValeur calculé
            INDArray qV = Nd4j.zeros(1, 1);
            qV.put(0, 0, newQValue);

            // train the neural network with inputs to match the QValue calculated previously
            model.fit(qSaving, qV);
        }else{
            // qSaving enregistre les inputs du dernier move
            qSaving = Nd4j.zeros(1,7);
        }

        int index = moveToIndex(nextMove);
        qSaving = initializeINDarray(pillsEatenInput, powerPillInput, pillInput[index], ghostInput[index],
                    ghostAfraidInput[index], entrapementInput[index], isGoingTowards[index]);

        // Save the data in the file data.txt
        //saveDataInFile(data);

        // netx move found
        myMove = nextMove;

        // save the NN in its file
        saveNN(model);

        // number of moves in the game +1 for the data save
        nbMoves += 1;

        // return the move
        return myMove;
    }

    private INDArray initializeINDarray(double pillsEaten, double powerPill, double pill, double ghost,
                                        double ghostAfraid, double entrapement, int isGoingTowards){
        INDArray array = Nd4j.zeros(1,7);
        array.put(0,0, pillsEaten);
        array.put(0,1, powerPill);
        array.put(0,2, pill);
        array.put(0,3, ghost);
        array.put(0,4, ghostAfraid);
        array.put(0,5, entrapement);
        array.put(0,6, isGoingTowards);

        return array;
    }

    private void saveNN(MultiLayerNetwork model){
        try {
            ModelSerializer.writeModel(model, "network.txt", false);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    /** Function that loads the neural network and returns it
     * if there is a problem with the loading, initialize a new one if
     * createNewNN == true
     * **/
    private MultiLayerNetwork loadNN(boolean createNewNN){
        MultiLayerNetwork model = null;
        try{
            model = ModelSerializer.restoreMultiLayerNetwork("network.txt");
        } catch (IOException e) {
            e.printStackTrace();
            if(createNewNN) {
                model = initializeModel(7, 1, 50, 0.0005);
                model.init();
            }
        }
        return model;
    }

    /** Save data in its file **/
    private void saveDataInFile(String d){
        BufferedWriter bw = null;
        FileWriter fw = null;

        try {
            fw = new FileWriter("data.txt", true);
            bw = new BufferedWriter(fw);
            bw.write(d);
            //System.out.println(d);
        } catch (IOException e) {

            e.printStackTrace();

        }finally {
            try {
                if (bw != null)
                    bw.close();
                if (fw != null)
                    fw.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }


    }

    /** Create a Neural Network and return it
     * 3 layers:
     *      7 neurons in input
     *      50 neurons in the hidden layer
     *      1 neuron in the output
     **/
    private MultiLayerNetwork initializeModel(int nbInputs, int nbOutputs, int nbNeuronsHidden, double learningRate){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(nbInputs).nOut(nbNeuronsHidden)
                            .weightInit(WeightInit.XAVIER)
                            .activation(Activation.RELU)
                            .build())
                .layer(1, new DenseLayer.Builder().nIn(nbNeuronsHidden).nOut(nbNeuronsHidden)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SIGMOID).weightInit(WeightInit.XAVIER)
                        .nIn(nbNeuronsHidden).nOut(nbOutputs).build())
                .pretrain(false).backprop(true).build();

        return new MultiLayerNetwork(conf);
    }

    /** Algos utilisés pour calculer les paramètres du réseau de neurones **/

    /** Algo 1 : paramètre par rapport au nombre de pills restantes **/
    public double PillsEatenInput(Game game) {
    	
    		//Nombre total de pills
    		double npills = game.getPillIndices().length;
    		//Nombre de pills restants
    		double nactivepills=game.getNumberOfActivePills();
 
    		return (npills-nactivepills)/npills;
    	
    }
    
    /** Algo 2 : paramètre par rapport à la durée restante de la powerpill **/
    public double powerPillInput(Game game) {

        double timeLeft=0;

        int nghosts = 0;
        for (GHOST type:GHOST.values()) {
            if (game.getGhostEdibleTime(type) > -1){ // the ghost is in sight
                nghosts += 1;
                timeLeft+=game.getGhostEdibleTime(type);
            }
        }


        if(nghosts == 0){ //cas 1 : powerpill inefficace à la connaissance de PacMan (elle ne sait pas pour les ghosts hors de son champs de vision)
            return 0;
        }
        else {
            timeLeft /= nghosts;
            return (Constants.EDIBLE_TIME - timeLeft)/Constants.EDIBLE_TIME;
        }

    }

    /** Algo 3 : paramètre pour la pill la plus proche dans chaque direction **/
    public double[] pillInput(Game game) {
        double[] closestPills = {longest_path, longest_path,longest_path,longest_path};
        int current=game.getPacmanCurrentNodeIndex();

        for(MOVE move : MOVE.values()) {
            int index = moveToIndex(move);
            if(index > -1) {
                double distance = longest_path;
                double temp;

                for (int i : game.getActivePillsIndices()) {
                    if (i > -1) {
                        temp = game.getShortestPathDistance(current, i, move);
                        if (temp < distance) {
                            distance = temp;
                        }
                    }
                }
                closestPills[moveToIndex(move)] = distance;
            }
        }

        for(int direction=0; direction<closestPills.length; direction++){
            closestPills[direction] = (longest_path-closestPills[direction])/longest_path;
        }

        return closestPills;
    }


    // cette fonction est une fonction de secours, au cas où celle au dessus ne fonctionnerait pas correctement (à cause dugetShortestPathDistance avec move en paramètre)
    public double[] pillInput2(Game game) {
        double[] closestPills = {longest_path, longest_path,longest_path,longest_path};
        int current_x = game.getNodeXCood(game.getPacmanCurrentNodeIndex());
        int current_y = game.getNodeYCood(game.getPacmanCurrentNodeIndex());
        int current=game.getPacmanCurrentNodeIndex();
        int current_i_distance;

        for(int i:game.getActivePillsIndices()) {
            if(i>-1) {
                current_i_distance=game.getShortestPathDistance(current, i);
                int i_x = game.getNodeXCood(i);
                int i_y = game.getNodeYCood(i);

                boolean up = (i_x == current_x && current_y-i_y > 0);
                boolean down = (i_x == current_x && current_y-i_y < 0);
                boolean left = (current_x-i_x > 0 && current_y-i_y == 0);
                boolean right = (current_x-i_x < 0 && current_y-i_y == 0);
                boolean none = !(up || down || left || right);
                if (!none){

                    int direction = -1;

                    if(up){
                        direction = 0;
                    }
                    if(down){
                        direction = 1;
                    }
                    if(left){
                        direction = 2;
                    }
                    if(right){
                        direction = 3;
                    }

                    if(current_i_distance < closestPills[direction]) {
                        closestPills[direction]=current_i_distance;
                    }
                }
            }
        }
        // normalization
        for (int direction=0; direction<closestPills.length; direction++){
            closestPills[direction] = (longest_path-closestPills[direction])/longest_path;
        }

        return closestPills;
    }


    /** Algo 4 : danger associé à chaque action **/
    public double[] ghostInput(Game game) {
        double a=longest_path, d= 0;
        double v = 0.8; 
        double[] danger = new double[4];

        // pour chaque direction, déterminer l'intersection la plus proche de Ms PacMan et la distance entre cette intersection et le fantôme qui en est le plus proche
        for (MOVE move: MOVE.values()) {
            int direction = moveToIndex(move);
            double b;
            if(direction != -1) {
                int junction_index = getNearestJunction(game, move, longest_path); // intersection la plus proche dans la direction move

                if (junction_index > -1) // intersection existante dans la direction move
                {
                    d = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), junction_index, move); // distance de l'intersection la plus proche dans la direction move

                    int ghost_index = getNearestGhostIndex(game, move, longest_path); // trouver le fantôme le plus proche dans la direction move
                    if(ghost_index > -1) { // si un fantôme existe
                        b = game.getShortestPathDistance(ghost_index, junction_index, move); // distance la plus courte entre le fantôme le plus proche et l'intersection la plus proche de Ms Pacman
                    }
                    else{ // pas de fantômes dans la direction move
                        b = longest_path;
                    }
                }
                else{ //si pas d'intersections de ce côté (cul-de-sac)
                    d = 0;
                    int ghost_index = getNearestGhostIndex(game, move, longest_path);
                    if (ghost_index == -1){ // pas de fantômes
                        b = longest_path;
                    }
                    else{
                        b = (double)game.getShortestPathDistance(ghost_index, game.getPacmanCurrentNodeIndex(), move)/2;
                    }
                }

                danger[direction] = (a + d * v - b) / a;
            }

        }

        return danger;
    	
    }

    /** Algo 5 : Nearest eatable ghost **/
    public double[] ghostAfraidInput(Game game) {
        double[] ghosts_afraid = new double[4];
        double a = longest_path;
        int ghostIndex=0;

        for (MOVE move : MOVE.values()) {

            int index = moveToIndex(move);
            if (index>-1) {
                double b = longest_path;
                int temp;

                for (GHOST type : GHOST.values()) {
                    ghostIndex = game.getGhostCurrentNodeIndex(type);
                    if (ghostIndex > -1 && game.isGhostEdible(type)) {
                        temp = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), ghostIndex, move);
                        if (temp > -1 && temp < b) {
                            b = temp;
                        }
                    }
                }

                ghosts_afraid[index] = b;
            }
        }
        //normalization
        for (int direction=0; direction<ghosts_afraid.length; direction++){
            ghosts_afraid[direction] = (a-ghosts_afraid[direction])/a;
        }

        return ghosts_afraid;

    }

    /**Algo 6 : cas où Ms PacMan doit se rapprocher d'un fantôme pour pouvoir s'enfuir (intersection entre eux) **/
    public double[] entrapmentInput(Game game) {
    	
        /*Pour cela, un algorithme liste toutes les intersections existantes
        , et pour chacune d’entre elles,
        l’algo évalue le temps nécessaire à Ms PacMan et à chaque fantôme,
        pour atteindre cette intersection.
        Une liste des intersections atteignables par Ms PacMan avant
        tous les autres fantômes est ensuite établie,
        et BFS est utilisé pour trouver tous les chemins les plus sûrs.
        L’algo retourne alors le pourcentage des routes sûres qui ne sont
        pas dans cette direction (permet de mesurer le danger de cette direction).
        EntrapmentInput(c) = (a-b(c))/a
        a = nombre total de routes sûres
        b(c) = nombre de routes sûres dans la direction c*/

        // récupérer la liste des intersections et des fantômes les plus proches dans chaque direction
        int[] junction_nodes = new int[4];
        int[] ghosts_index = new int[4];
        for (MOVE move : MOVE.values()){
            int direction = moveToIndex(move);
            if(direction>-1)
            {
                junction_nodes[direction] = getNearestJunction(game, move, longest_path);
                ghosts_index[direction] = getNearestGhostIndex(game, move, longest_path);
            }
        }

        // filtrer les intersections pour ne garder que celles atteignables par Ms PacMan avant les fantômes,
        // on ne considère qu'un seul chemin par intersection (de toutes façons on ne peut pas tous les calculer
        // car on a une visibilité partielle)
        double[] safe_paths = new double[4];
        int nsafepaths = 0;
        for (int i=0; i<junction_nodes.length; i++){
            int junction = junction_nodes[i];
            double pacman_junction_distance = longest_path;
            double ghost_junction_distance = longest_path;
            if(junction > -1) // il y a une intersection
            {
                pacman_junction_distance = game.getShortestPathDistance(game.getPacmanCurrentNodeIndex(), junction, indexToMove(i));
                if(ghosts_index[i] > -1){
                    ghost_junction_distance = game.getShortestPathDistance(ghosts_index[i], junction);
                }
            }
            if(ghost_junction_distance > pacman_junction_distance){
                safe_paths[i] = 1;
                nsafepaths ++;
            }
            else{ // cas ou pas d'intersections compris
                safe_paths[i] = 0;
            }


        }

        if(nsafepaths>0) {
            for (int i = 0; i < safe_paths.length; i++) {
                safe_paths[i] /= nsafepaths;
            }
        }
        return safe_paths;
    }
    
    /**Algo 7 Moving Direction**/
    public int[] isGoingToward() {
        /* return a list of boolean, with 1 if PacMan is already moving in the direction
        * and 0 if not
        */
        int[] moving_direction = new int[4];

        for(MOVE move : MOVE.values()){
            int direction = moveToIndex(move);
            if(direction != -1){
                moving_direction[direction] = (move==this.getMove()) ? 1 : 0;
            }
        }

        return moving_direction;
    }
    
}