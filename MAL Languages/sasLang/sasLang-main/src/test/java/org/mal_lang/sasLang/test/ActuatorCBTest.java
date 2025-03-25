package org.mal_lang.saslang.test;

import core.Attacker;
import core.AttackStep;
import org.junit.jupiter.api.Test;

public class ActuatorCBTest extends SasLangTest {
    private static class ActuatorCBTestModel {
        public final ActuatorCB actuatorCB = new
            ActuatorCB("actuatorCB");
        public final CircuitBreaker circuitBreaker = new
            CircuitBreaker("circuitBreaker");

        public ActuatorCBTestModel() {
            actuatorCB.addCircuitBreaker(circuitBreaker);
        }

        public void addAttacker(Attacker attacker, AttackStep attackpoint) {
            attacker.addAttackPoint(attackpoint);
        }
    }

    @Test
    public void testActuatorCBManipulate() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new ActuatorCBTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.actuatorCB.manipulate);
        attacker.attack();

        model.circuitBreaker.manipulationOfControl.assertCompromisedInstantaneously();
    }

    @Test
    public void testActuatorCBBlock() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new ActuatorCBTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.actuatorCB.block);
        attacker.attack();

        model.circuitBreaker.lossOfControl.assertCompromisedInstantaneously();
    }
}
