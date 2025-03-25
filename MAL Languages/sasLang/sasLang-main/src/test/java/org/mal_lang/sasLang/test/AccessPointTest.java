package org.mal_lang.saslang.test;

import core.Attacker;
import core.AttackStep;
import org.junit.jupiter.api.Test;

public class AccessPointTest extends SasLangTest {
    private static class AccessPointTestModel {
        public final AccessPoint accessPoint = new
            AccessPoint("accesspoint", false, false);
        public final IEDRouter iedRouter = new
            IEDRouter("iedrouter", false, false, false, false, false, false);

        public AccessPointTestModel() {
            iedRouter.addAccessPoint(accessPoint);
        }

        public void addAttacker(Attacker attacker, AttackStep attackpoint) {
            attacker.addAttackPoint(attackpoint);
        }
    }

    @Test
    public void testAccessPointLogicalConnection() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new AccessPointTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.accessPoint.logicalConnection);
        attacker.attack();

        model.iedRouter.logicalConnection.assertCompromisedInstantaneously();
    }
}
