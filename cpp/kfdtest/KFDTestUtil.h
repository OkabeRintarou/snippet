#pragma once

#include <gtest/gtest.h>
#include <hsakmt.h>

#define ASSERT_SUCCESS(_val) ASSERT_EQ(HSAKMT_STATUS_SUCCESS, _val)
#define EXPECT_SUCCESS(_val) EXPECT_EQ(HSAKMT_STATUS_SUCCESS, _val)