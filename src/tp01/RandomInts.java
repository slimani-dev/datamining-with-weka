package tp01;

import java.util.Random;

public class RandomInts {
    public static void main(String[] args) {
        int number = 10;

        Random random = new Random(1);
        int[] ints = random.ints(number).toArray();

        for (int i = 0; i < ints.length; i++) {
            System.out.println(Math.abs(ints[i]) % ints.length);

            ints[i] = Math.abs(ints[i]) % ints.length;
        }

        System.out.println("not in");

        for (int i = 0; i < ints.length; i++) {
            boolean has = false;

            for (int j = 0; j < ints.length; j++) {
                if(ints[j] == i) {
                    has = true;
                    break;
                }
            }

            if (!has) {
                System.out.println(i);
            }
        }

    }
}
