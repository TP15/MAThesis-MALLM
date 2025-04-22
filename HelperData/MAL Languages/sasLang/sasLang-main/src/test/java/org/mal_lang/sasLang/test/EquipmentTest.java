package org.mal_lang.saslang.test;

import core.Attacker;
import core.AttackStep;
import org.junit.jupiter.api.Test;

public class EquipmentTest extends SasLangTest {
    private static class EquipmentTestModel {
        public final Equipment equipment = new
            Equipment("equipment");

        public void addAttacker(Attacker attacker, AttackStep attackpoint) {
            attacker.addAttackPoint(attackpoint);
        }
    }

    @Test
    public void testEquipmentLossOfSafety() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.lossOfSafety);
        attacker.attack();

        model.equipment.safetyMechanismsOffline.assertCompromisedInstantaneously();
        model.equipment.attemptPreemptiveShutdownOnSafetyLoss.assertCompromisedInstantaneously();
    }

    @Test
    public void testEquipmentShutdown() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.shutdown);
        attacker.attack();

        model.equipment.lossOfAvailability.assertCompromisedInstantaneously();
    }
   @Test
    public void testEquipmentDamageToProperty() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.damageToProperty);
        attacker.attack();

        model.equipment.shutdown.assertCompromisedInstantaneously();        
	model.equipment.lossOfProductivityAndRevenue.assertCompromisedInstantaneously();
    }
   @Test
    public void testEquipmentLossOfControl() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.lossOfControl);
        attacker.attack();

        model.equipment.attemptPreemptiveShutdown.assertCompromisedInstantaneously();        
    }
   @Test
    public void testEquipmentLossOfView() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.lossOfView);
        attacker.attack();

        model.equipment.attemptPreemptiveShutdown.assertCompromisedInstantaneously();        
    }
   @Test
    public void testEquipmentLossOfAvailability() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.lossOfAvailability);
        attacker.attack();

        model.equipment.lossOfProductivityAndRevenue.assertCompromisedInstantaneously();        
    }
   @Test
    public void testEquipmentManipulationOfControl() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.manipulationOfControl);
        attacker.attack();

        model.equipment.unsafeState.assertCompromisedInstantaneously();        
    }
   @Test
    public void testEquipmentManipulationOfView() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new EquipmentTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.equipment.manipulationOfView);
        attacker.attack();

        model.equipment.unsafeState.assertCompromisedInstantaneously();        
    }

}
