import entrants.pacman.CBYHJJ.PacManPlayer;
import examples.StarterGhostComm.Blinky;
import examples.StarterGhostComm.Inky;
import examples.StarterGhostComm.Pinky;
import examples.StarterGhostComm.Sue;
import pacman.Executor;
import pacman.controllers.IndividualGhostController;
import pacman.controllers.MASController;
import pacman.game.Constants.*;

import java.util.EnumMap;

public class Play {
    public static void main(String[] args) {

        // Load the game with visual
        Executor executor = new Executor.Builder()
                .setVisual(true)
                .setTickLimit(4000)
                .build();

        // Initialize the Ghosts
        EnumMap<GHOST, IndividualGhostController> controllers = new EnumMap<>(GHOST.class);

        controllers.put(GHOST.INKY, new Inky());
        controllers.put(GHOST.BLINKY, new Blinky());
        controllers.put(GHOST.PINKY, new Pinky());
        controllers.put(GHOST.SUE, new Sue());

        // Load the PacMan class to play
        PacManPlayer pacman = new PacManPlayer(0.5);
        // Load controller
        MASController msControl = new MASController(controllers);

        // Launch the game
        executor.runGameTimed(pacman, msControl);

    }
}
