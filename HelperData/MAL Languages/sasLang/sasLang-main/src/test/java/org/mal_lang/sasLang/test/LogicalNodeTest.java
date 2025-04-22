package org.mal_lang.saslang.test;

import core.Attacker;
import core.AttackStep;
import org.junit.jupiter.api.Test;

public class LogicalNodeTest extends SasLangTest {
    private static class LogicalNodeTestModel {
        public final LogicalNode logicalNode = new
            LogicalNode("logicalNode");
        public final Equipment equipment = new
            Equipment("equipment");
        public final Actuator actuator = new 
	    Actuator("actuator");

        public LogicalNodeTestModel() {
            logicalNode.addEquipment(equipment);
            logicalNode.addActuator(actuator);
        }

        public void addAttacker(Attacker attacker, AttackStep attackpoint) {
            attacker.addAttackPoint(attackpoint);
        }
    }

    @Test
    public void testLogicalNodeManipulationOfControl() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new LogicalNodeTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.logicalNode.manipulationOfControl);
        attacker.attack();

        model.equipment.manipulationOfControl.assertCompromisedInstantaneously();
        model.actuator.manipulate.assertCompromisedInstantaneously();
    }

    @Test
    public void testLogicalNodeLossOfControl() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new LogicalNodeTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.logicalNode.lossOfControl);
        attacker.attack();

        model.equipment.manipulationOfControl.assertCompromisedInstantaneously();
        model.actuator.block.assertCompromisedInstantaneously();
    }

}
