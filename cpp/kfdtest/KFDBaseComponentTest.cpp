#include "KFDBaseComponentTest.h"
#include "KFDTestUtil.h"

void KFDBaseComponentTest::SetUp() {
    ASSERT_SUCESS(hsaKmtOpenKFD());
    EXPECT_SUCCESS(hsaKmtGetVersion(&m_VersionInfo));

    memset(&m_SystemProperties, 0, sizeof(m_SystemProperties));
    memset(m_RenderNodes, 0, sizeof(m_RenderNodes));

    ASSERT_SUCESS(hsaKmtAcquireSystemProperties(&m_SystemProperties));
    ASSERT_GT(m_SystemProperties.NumNodes, HSAuint32(0));

}

void KFDBaseComponentTest::TearDown() {
    ASSERT_SUCESS(hsaKmtCloseKFD());
}

TEST_F(KFDBaseComponentTest, FirstTest) {

}
