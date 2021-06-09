import entrants.pacman.CBYHJJ.PacManTraining;
import examples.StarterGhostComm.Blinky;
import examples.StarterGhostComm.Inky;
import examples.StarterGhostComm.Pinky;
import examples.StarterGhostComm.Sue;
import pacman.Executor;
import pacman.controllers.IndividualGhostController;
import pacman.controllers.MASController;
import pacman.game.Constants.*;

import java.util.EnumMap;

/**
 * Created by pwillic on 06/05/2016.
 */
public class Training {
	
    public static void main(String[] args) {

        // Load the game with no visual
        Executor executor = new Executor.Builder()
                .setVisual(false)
                .setTickLimit(4000)
                .build();

        // Initialize the Ghosts
        EnumMap<GHOST, IndividualGhostController> controllers = new EnumMap<>(GHOST.class);

        controllers.put(GHOST.INKY, new Inky());
        controllers.put(GHOST.BLINKY, new Blinky());
        controllers.put(GHOST.PINKY, new Pinky());
        controllers.put(GHOST.SUE, new Sue());

        // Load the PacMan Class to train
        PacManTraining pacman;

        // Train on 7000 games
        int i = 2650;
        while (i < 7000) {
            System.out.println("Partie "+i);

            // Initialize PacMan and the controller
            pacman = new PacManTraining(i);
            MASController msControl = new MASController(controllers);

            // Play the game without showing it
            executor.runGameTimed(pacman, msControl);

            i++;
        }
    }
}
