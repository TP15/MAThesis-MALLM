package org.mal_lang.saslang.test;

import core.Attacker;
import core.AttackStep;
import org.junit.jupiter.api.Test;

public class IEDRouterTest extends SasLangTest {
    private static class IEDRouterTestModel {
        public final IEDRouter iedRouter = new
            IEDRouter("iedRouter");
        public final AccessPoint accessPoint = new
            AccessPoint("accessPoint");

        public IEDRouterTestModel() {
            iedRouter.addAccessPoint(accessPoint);
        }

        public void addAttacker(Attacker attacker, AttackStep attackpoint) {
            attacker.addAttackPoint(attackpoint);
        }
    }

    @Test
    public void testIEDRouterLogicalConnection() {
        printTestName(Thread.currentThread().getStackTrace()[1].getMethodName());
        var model = new IEDRouterTestModel();

        var attacker = new Attacker();
        model.addAttacker(attacker,model.iedRouter.logicalConnection);
        attacker.attack();

        model.accessPoint.attemptConnectToApplications.assertCompromisedInstantaneously();
    }

}
