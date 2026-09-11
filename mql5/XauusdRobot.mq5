//+------------------------------------------------------------------+
//|                                               XauusdRobot.mq5     |
//|   XAUUSD Blueprint v1.1 -- combination F, as validated in Python  |
//+------------------------------------------------------------------+
//
// A compiled port of the Python research implementation in this repository.
// The point of the port is deployment: a .ex5 runs inside the terminal, so it
// survives on MetaTrader's own Virtual Hosting and needs no external process
// supervising it.
//
// The rules are identical to the Python engine and must stay that way -- the
// Python side remains the reference, and scripts/compare_mql5_signals.py checks
// the two agree on the same history.
//
// Differences from blueprint v1.1, each measured rather than assumed:
//   * Regime filter is H4/H1/M30/M5 (D1 and M15 removed; M15 was byte-identical
//     redundancy, D1 was removed by instruction).
//   * Combination F: OB structure break not required, WPR lead 5 bars, setup
//     expiry 8 bars, S/R zone 0.15 ATR, push body ratio 0.65.
//
// Everything decides on CLOSED bars only. Shift 1 is the last closed M5 bar;
// shift 0 is still forming and is never read.
//
#property copyright "XAUUSD Robot v1.1"
#property version   "1.10"
#property strict

#include <Trade\Trade.mqh>

//--- Section 13 configuration -------------------------------------------------
input group "=== Risk ==="
input double InpRiskPercent        = 2.0;    // % of INITIAL balance per trade
input double InpRewardRisk         = 3.0;    // take profit as a multiple of risk

input group "=== Regime filter (EMA200) ==="
input int    InpEmaPeriod          = 200;
input bool   InpUseH4              = true;
input bool   InpUseH1              = true;
input bool   InpUseM30             = true;
input bool   InpUseM5              = true;   // load-bearing: see README

input group "=== Momentum ==="
input int    InpWprPeriod          = 49;
input double InpWprOversold        = -80.0;
input double InpWprOverbought      = -20.0;
input int    InpWprMaxLeadBars     = 5;      // combination F (blueprint: 2)

input group "=== Zones ==="
input int    InpAtrPeriod          = 14;
input double InpObDisplacementAtr  = 1.0;
input int    InpObDisplacementBars = 3;
input bool   InpObRequireSwingBrk  = false;  // combination F (blueprint: true)
input int    InpZoneMaxAgeBars     = 96;
input int    InpSrLookback         = 20;
input int    InpSrPivotLeft        = 2;
input int    InpSrPivotRight       = 2;
input double InpSrZoneAtr          = 0.15;   // combination F (blueprint: 0.10)
input double InpFvgMinAtr          = 0.10;

input group "=== Setup lifecycle ==="
input int    InpReactionMaxBars    = 2;
input int    InpSetupExpiryBars    = 8;      // combination F (blueprint: 5)
input double InpPushBodyRatio      = 0.65;   // combination F (blueprint: 0.60)
input bool   InpPush2BreaksPush1   = true;
input int    InpPushCandlesReq     = 2;      // 2 is where the edge is

input group "=== Stops ==="
input double InpSlBufferAtr        = 0.10;
input double InpSlBufferSpreadMult = 2.0;

input group "=== Circuit breakers (Section 5.5) ==="
input int    InpMaxTradesPerDay    = 3;
input int    InpMaxLossesPerDay    = 2;
input double InpDailyLossLimitR    = -2.0;
input double InpMaxPeakDrawdown    = 0.15;
input double InpMaxSpreadVsSl      = 0.10;
input double InpMaxSpreadAtr       = 0.15;
input int    InpCooldownBars       = 3;
input double InpTargetMultiplier   = 10.0;

input group "=== Execution ==="
input ulong  InpMagic              = 20260910;
input ulong  InpDeviationPoints    = 20;
input bool   InpDryRun             = true;   // must be set false deliberately
input bool   InpVerboseLog         = true;

//--- constants ---------------------------------------------------------------
#define ZONE_OB  0
#define ZONE_FVG 1
#define ZONE_SR  2
#define DIR_BUY   1
#define DIR_SELL -1
#define REG_MIXED 0

#define ST_IDLE      0
#define ST_ARMED     1
#define ST_WPR       2
#define ST_PUSH1     3

//--- types -------------------------------------------------------------------
struct Zone
  {
   int      id;
   int      type;
   int      direction;
   double   low;
   double   high;
   datetime created;
   bool     touched;
   datetime touch_time;
   bool     retired;
  };

struct Setup
  {
   bool     active;
   int      direction;
   int      zone_type;
   double   zone_low;
   double   zone_high;
   int      confluence;
   datetime reaction_time;
   datetime expiry_time;
   double   reaction_extreme;
   bool     wpr_extreme_seen;
   bool     wpr_confirmed;
   bool     push1_set;
   datetime push1_time;
   double   push1_high;
   double   push1_low;
   int      state;
  };

//--- globals -----------------------------------------------------------------
CTrade   trade;
Zone     g_zones[];
Setup    g_setup;
int      g_next_zone_id = 1;
datetime g_last_bar     = 0;

int      h_ema_h4 = INVALID_HANDLE, h_ema_h1 = INVALID_HANDLE;
int      h_ema_m30 = INVALID_HANDLE, h_ema_m5 = INVALID_HANDLE;
int      h_atr = INVALID_HANDLE, h_wpr = INVALID_HANDLE;

// Persisted safety state (GlobalVariables survive terminal restarts).
double   g_initial_balance = 0.0;
double   g_equity_peak     = 0.0;
bool     g_drawdown_locked = false;
bool     g_target_reached  = false;
int      g_trades_today    = 0;
int      g_losses_today    = 0;
double   g_daily_r         = 0.0;
datetime g_broker_day      = 0;
datetime g_cooldown_until  = 0;
double   g_open_risk_money = 0.0;

string   GV(const string key) { return StringFormat("XR_%s_%I64u_%s", _Symbol, InpMagic, key); }

//+------------------------------------------------------------------+
//| Persistence                                                      |
//+------------------------------------------------------------------+
void SaveState()
  {
   GlobalVariableSet(GV("init_bal"),   g_initial_balance);
   GlobalVariableSet(GV("peak"),       g_equity_peak);
   GlobalVariableSet(GV("dd_lock"),    g_drawdown_locked ? 1 : 0);
   GlobalVariableSet(GV("target"),     g_target_reached ? 1 : 0);
   GlobalVariableSet(GV("trades"),     g_trades_today);
   GlobalVariableSet(GV("losses"),     g_losses_today);
   GlobalVariableSet(GV("daily_r"),    g_daily_r);
   GlobalVariableSet(GV("day"),        (double)g_broker_day);
   GlobalVariableSet(GV("cooldown"),   (double)g_cooldown_until);
   GlobalVariableSet(GV("open_risk"),  g_open_risk_money);
  }

void LoadState()
  {
   if(GlobalVariableCheck(GV("init_bal")))
     {
      g_initial_balance = GlobalVariableGet(GV("init_bal"));
      g_equity_peak     = GlobalVariableGet(GV("peak"));
      g_drawdown_locked = GlobalVariableGet(GV("dd_lock")) > 0.5;
      g_target_reached  = GlobalVariableGet(GV("target")) > 0.5;
      g_trades_today    = (int)GlobalVariableGet(GV("trades"));
      g_losses_today    = (int)GlobalVariableGet(GV("losses"));
      g_daily_r         = GlobalVariableGet(GV("daily_r"));
      g_broker_day      = (datetime)GlobalVariableGet(GV("day"));
      g_cooldown_until  = (datetime)GlobalVariableGet(GV("cooldown"));
      g_open_risk_money = GlobalVariableGet(GV("open_risk"));
      PrintFormat("state restored: initial %.2f, peak %.2f, dd_lock=%s target=%s",
                  g_initial_balance, g_equity_peak,
                  g_drawdown_locked ? "yes" : "no", g_target_reached ? "yes" : "no");
     }
   else
     {
      g_initial_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_equity_peak     = g_initial_balance;
      PrintFormat("new state: initial balance reference %.2f", g_initial_balance);
      SaveState();
     }
  }

//+------------------------------------------------------------------+
void Log(const string msg)
  {
   if(InpVerboseLog)
      Print(msg);
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpDeviationPoints);
   trade.SetTypeFillingBySymbol(_Symbol);

   h_ema_h4  = iMA(_Symbol, PERIOD_H4,  InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_h1  = iMA(_Symbol, PERIOD_H1,  InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_m30 = iMA(_Symbol, PERIOD_M30, InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   h_ema_m5  = iMA(_Symbol, PERIOD_M5,  InpEmaPeriod, 0, MODE_EMA, PRICE_CLOSE);
   h_atr     = iATR(_Symbol, PERIOD_M5, InpAtrPeriod);
   h_wpr     = iWPR(_Symbol, PERIOD_M5, InpWprPeriod);

   if(h_ema_h4 == INVALID_HANDLE || h_ema_h1 == INVALID_HANDLE ||
      h_ema_m30 == INVALID_HANDLE || h_ema_m5 == INVALID_HANDLE ||
      h_atr == INVALID_HANDLE || h_wpr == INVALID_HANDLE)
     {
      Print("ERROR: could not create indicator handles");
      return(INIT_FAILED);
     }

   ArrayResize(g_zones, 0);
   ZeroSetup();
   LoadState();

   PrintFormat("XauusdRobot v1.1 on %s | mode: %s | risk %.2f%% | regime %s%s%s%s",
               _Symbol, InpDryRun ? "DRY RUN" : "LIVE ORDERS", InpRiskPercent,
               InpUseH4 ? "H4 " : "", InpUseH1 ? "H1 " : "",
               InpUseM30 ? "M30 " : "", InpUseM5 ? "M5" : "");
   if(!InpDryRun)
      Print("*** LIVE ORDER PLACEMENT ENABLED ***");
   ReconcilePosition();
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   SaveState();
   IndicatorRelease(h_ema_h4);  IndicatorRelease(h_ema_h1);
   IndicatorRelease(h_ema_m30); IndicatorRelease(h_ema_m5);
   IndicatorRelease(h_atr);     IndicatorRelease(h_wpr);
  }

//+------------------------------------------------------------------+
void ZeroSetup()
  {
   g_setup.active           = false;
   g_setup.direction        = 0;
   g_setup.zone_type        = -1;
   g_setup.zone_low         = 0;
   g_setup.zone_high        = 0;
   g_setup.confluence       = 0;
   g_setup.reaction_time    = 0;
   g_setup.expiry_time      = 0;
   g_setup.reaction_extreme = 0;
   g_setup.wpr_extreme_seen = false;
   g_setup.wpr_confirmed    = false;
   g_setup.push1_set        = false;
   g_setup.push1_time       = 0;
   g_setup.push1_high       = 0;
   g_setup.push1_low        = 0;
   g_setup.state            = ST_IDLE;
  }

//+------------------------------------------------------------------+
//| Indicator helpers -- all read CLOSED bars (shift >= 1)           |
//+------------------------------------------------------------------+
double BufferValue(const int handle, const int shift)
  {
   double buf[];
   if(CopyBuffer(handle, 0, shift, 1, buf) != 1)
      return(EMPTY_VALUE);
   return(buf[0]);
  }

double Atr(const int shift)  { return BufferValue(h_atr, shift); }
double Wpr(const int shift)  { return BufferValue(h_wpr, shift); }

//+------------------------------------------------------------------+
//| Section 3.1 -- multi-timeframe EMA200 regime on closed bars       |
//+------------------------------------------------------------------+
bool TimeframeSide(const ENUM_TIMEFRAMES tf, const int handle, int &side)
  {
   double ema = BufferValue(handle, 1);       // last CLOSED bar of that timeframe
   double close = iClose(_Symbol, tf, 1);
   if(ema == EMPTY_VALUE || ema <= 0 || close <= 0)
      return(false);
   side = (close > ema) ? DIR_BUY : ((close < ema) ? DIR_SELL : 0);
   return(true);
  }

int ComputeRegime()
  {
   int side, agreed = 0, used = 0;

   if(InpUseH4)
     {
      if(!TimeframeSide(PERIOD_H4, h_ema_h4, side)) return(REG_MIXED);
      used++; agreed += side;
     }
   if(InpUseH1)
     {
      if(!TimeframeSide(PERIOD_H1, h_ema_h1, side)) return(REG_MIXED);
      used++; agreed += side;
     }
   if(InpUseM30)
     {
      if(!TimeframeSide(PERIOD_M30, h_ema_m30, side)) return(REG_MIXED);
      used++; agreed += side;
     }
   if(InpUseM5)
     {
      if(!TimeframeSide(PERIOD_M5, h_ema_m5, side)) return(REG_MIXED);
      used++; agreed += side;
     }
   if(used == 0)
      return(REG_MIXED);
   // Strict alignment: every timeframe on the same side, no abstentions.
   if(agreed == used)  return(DIR_BUY);
   if(agreed == -used) return(DIR_SELL);
   return(REG_MIXED);
  }

//+------------------------------------------------------------------+
//| Zone helpers                                                     |
//+------------------------------------------------------------------+
double ZoneNear(const Zone &z) { return z.direction == DIR_BUY ? z.high : z.low; }
double ZoneFar(const Zone &z)  { return z.direction == DIR_BUY ? z.low  : z.high; }

void AddZone(const int type, const int direction, const double lo, const double hi,
             const datetime created)
  {
   int n = ArraySize(g_zones);
   ArrayResize(g_zones, n + 1);
   g_zones[n].id         = g_next_zone_id++;
   g_zones[n].type       = type;
   g_zones[n].direction  = direction;
   g_zones[n].low        = lo;
   g_zones[n].high       = hi;
   g_zones[n].created    = created;
   g_zones[n].touched    = false;
   g_zones[n].touch_time = 0;
   g_zones[n].retired    = false;
  }

void PurgeRetiredZones()
  {
   int n = ArraySize(g_zones), keep = 0;
   for(int i = 0; i < n; i++)
      if(!g_zones[i].retired)
        {
         if(keep != i) g_zones[keep] = g_zones[i];
         keep++;
        }
   ArrayResize(g_zones, keep);
  }

//+------------------------------------------------------------------+
//| Section 4.1 -- Fair Value Gap on the three newest closed bars     |
//+------------------------------------------------------------------+
void DetectFvg()
  {
   double atr = Atr(1);
   if(atr == EMPTY_VALUE || atr <= 0) return;
   double min_size = InpFvgMinAtr * atr;

   double c1_high = iHigh(_Symbol, PERIOD_M5, 3), c1_low = iLow(_Symbol, PERIOD_M5, 3);
   double c3_high = iHigh(_Symbol, PERIOD_M5, 1), c3_low = iLow(_Symbol, PERIOD_M5, 1);
   datetime t = iTime(_Symbol, PERIOD_M5, 1);

   if(c3_low > c1_high && (c3_low - c1_high) >= min_size)
      AddZone(ZONE_FVG, DIR_BUY, c1_high, c3_low, t);
   else if(c3_high < c1_low && (c1_low - c3_high) >= min_size)
      AddZone(ZONE_FVG, DIR_SELL, c3_high, c1_low, t);
  }

//+------------------------------------------------------------------+
//| Section 4.3 -- confirmed 2-left/2-right pivots become S/R zones   |
//+------------------------------------------------------------------+
bool IsPivotHigh(const int shift)
  {
   double v = iHigh(_Symbol, PERIOD_M5, shift);
   for(int k = 1; k <= InpSrPivotLeft; k++)
      if(v <= iHigh(_Symbol, PERIOD_M5, shift + k)) return(false);
   for(int k = 1; k <= InpSrPivotRight; k++)
      if(v <= iHigh(_Symbol, PERIOD_M5, shift - k)) return(false);
   return(true);
  }

bool IsPivotLow(const int shift)
  {
   double v = iLow(_Symbol, PERIOD_M5, shift);
   for(int k = 1; k <= InpSrPivotLeft; k++)
      if(v >= iLow(_Symbol, PERIOD_M5, shift + k)) return(false);
   for(int k = 1; k <= InpSrPivotRight; k++)
      if(v >= iLow(_Symbol, PERIOD_M5, shift - k)) return(false);
   return(true);
  }

void DetectSr()
  {
   double atr = Atr(1);
   if(atr == EMPTY_VALUE || atr <= 0) return;
   double half = InpSrZoneAtr * atr;

   // A pivot is only knowable once `right` further bars have closed, so the
   // newly confirmable pivot sits at shift 1 + right.
   int p = 1 + InpSrPivotRight;
   datetime t = iTime(_Symbol, PERIOD_M5, 1);

   if(IsPivotHigh(p))
     {
      double price = iHigh(_Symbol, PERIOD_M5, p);
      AddZone(ZONE_SR, DIR_SELL, price - half, price + half, t);
     }
   if(IsPivotLow(p))
     {
      double price = iLow(_Symbol, PERIOD_M5, p);
      AddZone(ZONE_SR, DIR_BUY, price - half, price + half, t);
     }
  }

//+------------------------------------------------------------------+
//| Section 4.2 -- Order Block: last opposite candle before impulse   |
//+------------------------------------------------------------------+
double LastConfirmedSwingHigh(const int from_shift)
  {
   for(int s = from_shift; s < from_shift + 200; s++)
      if(IsPivotHigh(s)) return(iHigh(_Symbol, PERIOD_M5, s));
   return(0.0);
  }

double LastConfirmedSwingLow(const int from_shift)
  {
   for(int s = from_shift; s < from_shift + 200; s++)
      if(IsPivotLow(s)) return(iLow(_Symbol, PERIOD_M5, s));
   return(0.0);
  }

void DetectOb()
  {
   double close_now = iClose(_Symbol, PERIOD_M5, 1);
   datetime t = iTime(_Symbol, PERIOD_M5, 1);

   // Candidate candles sit 1..InpObDisplacementBars bars before the current
   // closed bar; the displacement is measured to this bar's close.
   for(int back = 1; back <= InpObDisplacementBars; back++)
     {
      int k = 1 + back;                       // candidate shift
      double atr_k = Atr(k);
      if(atr_k == EMPTY_VALUE || atr_k <= 0) continue;

      double o = iOpen(_Symbol, PERIOD_M5, k), c = iClose(_Symbol, PERIOD_M5, k);
      double h = iHigh(_Symbol, PERIOD_M5, k), l = iLow(_Symbol, PERIOD_M5, k);
      double need = InpObDisplacementAtr * atr_k;

      if(c < o)                               // bearish candle -> bullish OB
        {
         if((close_now - c) >= need)
           {
            bool ok = true;
            if(InpObRequireSwingBrk)
              {
               double sw = LastConfirmedSwingHigh(1 + InpSrPivotRight);
               ok = (sw > 0.0 && close_now > sw);
              }
            if(ok) { AddZone(ZONE_OB, DIR_BUY, l, h, t); return; }
           }
        }
      else if(c > o)                          // bullish candle -> bearish OB
        {
         if((c - close_now) >= need)
           {
            bool ok = true;
            if(InpObRequireSwingBrk)
              {
               double sw = LastConfirmedSwingLow(1 + InpSrPivotRight);
               ok = (sw > 0.0 && close_now < sw);
              }
            if(ok) { AddZone(ZONE_OB, DIR_SELL, l, h, t); return; }
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Confluence: overlapping same-direction zones of other types       |
//+------------------------------------------------------------------+
int ConfluenceScore(const int idx)
  {
   bool seen_ob = false, seen_fvg = false, seen_sr = false;
   for(int i = 0; i < ArraySize(g_zones); i++)
     {
      if(g_zones[i].retired) continue;
      if(g_zones[i].direction != g_zones[idx].direction) continue;
      if(g_zones[i].high < g_zones[idx].low || g_zones[i].low > g_zones[idx].high) continue;
      if(g_zones[i].type == ZONE_OB)  seen_ob  = true;
      if(g_zones[i].type == ZONE_FVG) seen_fvg = true;
      if(g_zones[i].type == ZONE_SR)  seen_sr  = true;
     }
   int score = (seen_ob ? 1 : 0) + (seen_fvg ? 1 : 0) + (seen_sr ? 1 : 0);
   return(MathMax(1, MathMin(3, score)));
  }

//+------------------------------------------------------------------+
//| Section 4.4 -- age, invalidate, detect touch and reaction         |
//+------------------------------------------------------------------+
void ProcessZones(const int regime)
  {
   double bar_high = iHigh(_Symbol, PERIOD_M5, 1);
   double bar_low  = iLow(_Symbol, PERIOD_M5, 1);
   double bar_close = iClose(_Symbol, PERIOD_M5, 1);
   datetime now = iTime(_Symbol, PERIOD_M5, 1);

   for(int i = 0; i < ArraySize(g_zones); i++)
     {
      if(g_zones[i].retired) continue;

      int age = iBarShift(_Symbol, PERIOD_M5, g_zones[i].created);

      // Untouched zones age out; touched ones live until invalidated.
      if(!g_zones[i].touched && age >= InpZoneMaxAgeBars)
        { g_zones[i].retired = true; continue; }

      // A close beyond the far boundary invalidates, touched or not.
      if(g_zones[i].direction == DIR_BUY && bar_close < ZoneFar(g_zones[i]))
        { g_zones[i].retired = true; continue; }
      if(g_zones[i].direction == DIR_SELL && bar_close > ZoneFar(g_zones[i]))
        { g_zones[i].retired = true; continue; }

      if(!g_zones[i].touched)
        {
         if(bar_high >= g_zones[i].low && bar_low <= g_zones[i].high)
           { g_zones[i].touched = true; g_zones[i].touch_time = now; }
         else
            continue;
        }

      int touch_age = iBarShift(_Symbol, PERIOD_M5, g_zones[i].touch_time);
      if(touch_age > InpReactionMaxBars)
        { g_zones[i].touched = false; g_zones[i].touch_time = 0; continue; }

      bool reacted = (g_zones[i].direction == DIR_BUY  && bar_close > ZoneNear(g_zones[i])) ||
                     (g_zones[i].direction == DIR_SELL && bar_close < ZoneNear(g_zones[i]));
      if(!reacted) continue;

      int score = ConfluenceScore(i);
      g_zones[i].retired = true;              // first reaction only

      if(!g_setup.active && g_zones[i].direction == regime && regime != REG_MIXED)
         ArmSetup(g_zones[i], score);
     }
   PurgeRetiredZones();
  }

//+------------------------------------------------------------------+
//| Section 7 -- setup state machine                                  |
//+------------------------------------------------------------------+
void ArmSetup(const Zone &z, const int score)
  {
   if(TimeCurrent() < g_cooldown_until) return;

   ZeroSetup();
   g_setup.active        = true;
   g_setup.direction     = z.direction;
   g_setup.zone_type     = z.type;
   g_setup.zone_low      = z.low;
   g_setup.zone_high     = z.high;
   g_setup.confluence    = score;
   g_setup.reaction_time = iTime(_Symbol, PERIOD_M5, 1);
   g_setup.expiry_time   = g_setup.reaction_time + InpSetupExpiryBars * PeriodSeconds(PERIOD_M5);
   g_setup.state         = ST_ARMED;

   int touch_shift = MathMax(1, iBarShift(_Symbol, PERIOD_M5, z.touch_time));
   double extreme = (z.direction == DIR_BUY) ? iLow(_Symbol, PERIOD_M5, 1)
                                             : iHigh(_Symbol, PERIOD_M5, 1);
   for(int s = 1; s <= touch_shift; s++)
     {
      if(z.direction == DIR_BUY) extreme = MathMin(extreme, iLow(_Symbol, PERIOD_M5, s));
      else                       extreme = MathMax(extreme, iHigh(_Symbol, PERIOD_M5, s));
     }
   g_setup.reaction_extreme = extreme;

   // The WPR extreme may begin up to InpWprMaxLeadBars before the first touch.
   int start = touch_shift + InpWprMaxLeadBars;
   for(int s = start; s >= 1; s--)
      AdvanceWpr(s);

   AdvancePush(1);
   Log(StringFormat("setup armed %s from %s zone [%.2f, %.2f] confluence %d",
                    g_setup.direction == DIR_BUY ? "BUY" : "SELL",
                    g_setup.zone_type == ZONE_OB ? "OB" : (g_setup.zone_type == ZONE_FVG ? "FVG" : "SR"),
                    g_setup.zone_low, g_setup.zone_high, score));
  }

void AdvanceWpr(const int shift)
  {
   double v = Wpr(shift);
   if(v == EMPTY_VALUE) return;

   if(g_setup.direction == DIR_BUY)
     {
      if(v <= InpWprOversold)
         g_setup.wpr_extreme_seen = true;
      else if(g_setup.wpr_extreme_seen && !g_setup.wpr_confirmed && v > InpWprOversold)
         g_setup.wpr_confirmed = true;
     }
   else
     {
      if(v >= InpWprOverbought)
         g_setup.wpr_extreme_seen = true;
      else if(g_setup.wpr_extreme_seen && !g_setup.wpr_confirmed && v < InpWprOverbought)
         g_setup.wpr_confirmed = true;
     }
  }

bool QualifyingPush(const int shift)
  {
   double o = iOpen(_Symbol, PERIOD_M5, shift), c = iClose(_Symbol, PERIOD_M5, shift);
   double h = iHigh(_Symbol, PERIOD_M5, shift), l = iLow(_Symbol, PERIOD_M5, shift);
   double range = h - l;
   if(range <= 0.0) return(false);
   if(MathAbs(c - o) / range < InpPushBodyRatio) return(false);
   return(g_setup.direction == DIR_BUY ? (c > o) : (c < o));
  }

// Returns true when the push requirement is satisfied on bar `shift`.
bool AdvancePush(const int shift)
  {
   bool q = QualifyingPush(shift);

   if(InpPushCandlesReq <= 1)
     {
      if(q)
        {
         g_setup.push1_set = true;
         g_setup.push1_time = iTime(_Symbol, PERIOD_M5, shift);
         return(true);
        }
      g_setup.push1_set = false;
      return(false);
     }

   // Two consecutive: Push 1 must be the immediately preceding bar.
   if(g_setup.push1_set && iBarShift(_Symbol, PERIOD_M5, g_setup.push1_time) == shift + 1)
     {
      if(q)
        {
         bool brk = true;
         if(InpPush2BreaksPush1)
            brk = (g_setup.direction == DIR_BUY)
                  ? (iClose(_Symbol, PERIOD_M5, shift) > g_setup.push1_high)
                  : (iClose(_Symbol, PERIOD_M5, shift) < g_setup.push1_low);
         if(brk) return(true);
        }
     }

   if(q)
     {
      g_setup.push1_set  = true;
      g_setup.push1_time = iTime(_Symbol, PERIOD_M5, shift);
      g_setup.push1_high = iHigh(_Symbol, PERIOD_M5, shift);
      g_setup.push1_low  = iLow(_Symbol, PERIOD_M5, shift);
     }
   else
      g_setup.push1_set = false;
   return(false);
  }

void UpdateSetup(const int regime)
  {
   if(!g_setup.active) return;

   double bar_close = iClose(_Symbol, PERIOD_M5, 1);
   double far = (g_setup.direction == DIR_BUY) ? g_setup.zone_low : g_setup.zone_high;

   if((g_setup.direction == DIR_BUY && bar_close < far) ||
      (g_setup.direction == DIR_SELL && bar_close > far))
     { Log("setup cancelled: zone invalidated"); ZeroSetup(); return; }

   if(iTime(_Symbol, PERIOD_M5, 1) > g_setup.expiry_time)
     { Log("setup expired"); ZeroSetup(); return; }

   // Frozen while the regime disagrees; it simply ages out if alignment
   // does not return (matches the blueprint pseudocode's early return).
   if(regime != g_setup.direction) return;

   if(g_setup.direction == DIR_BUY)
      g_setup.reaction_extreme = MathMin(g_setup.reaction_extreme, iLow(_Symbol, PERIOD_M5, 1));
   else
      g_setup.reaction_extreme = MathMax(g_setup.reaction_extreme, iHigh(_Symbol, PERIOD_M5, 1));

   AdvanceWpr(1);
   bool push_ready = AdvancePush(1);

   g_setup.state = g_setup.push1_set && g_setup.wpr_confirmed ? ST_PUSH1
                   : (g_setup.wpr_confirmed ? ST_WPR : ST_ARMED);

   if(push_ready && g_setup.wpr_confirmed)
      AttemptEntry();
  }

//+------------------------------------------------------------------+
//| Section 5 -- risk engine                                          |
//+------------------------------------------------------------------+
double MoneyPerLot(const double stop_distance)
  {
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0.0) return(0.0);
   return(stop_distance / tick_size * tick_value);
  }

double NormaliseVolume(double lots)
  {
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0) return(0.0);
   lots = MathFloor(lots / step) * step;      // round DOWN so risk is never exceeded
   if(lots < vmin) return(0.0);
   return(MathMin(lots, vmax));
  }

void AttemptEntry()
  {
   double atr = Atr(1);
   if(atr == EMPTY_VALUE || atr <= 0) { ZeroSetup(); return; }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = ask - bid;
   double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   double buffer = MathMax(InpSlBufferAtr * atr, InpSlBufferSpreadMult * spread);
   double entry, stop, target, stop_distance;

   if(g_setup.direction == DIR_BUY)
     {
      entry = ask;
      double base = MathMin(g_setup.zone_low, g_setup.reaction_extreme);
      stop = base - buffer;
      stop_distance = entry - stop;
      target = entry + InpRewardRisk * stop_distance;
     }
   else
     {
      entry = bid;
      double base = MathMax(g_setup.zone_high, g_setup.reaction_extreme);
      stop = base + buffer + spread;
      stop_distance = stop - entry;
      target = entry - InpRewardRisk * stop_distance;
     }

   if(stop_distance <= 0.0) { Reject("invalid_stop_distance"); return; }

   //--- Section 5.5 safety gates
   if(g_target_reached)   { Reject("target_reached"); return; }
   if(g_drawdown_locked)  { Reject("drawdown_locked"); return; }
   if(OpenPositions() > 0){ Reject("position_already_open"); return; }
   if(TimeCurrent() < g_cooldown_until) { Reject("cooldown"); return; }
   if(g_trades_today >= InpMaxTradesPerDay) { Reject("max_trades_today"); return; }
   if(g_losses_today >= InpMaxLossesPerDay) { Reject("max_losses_today"); return; }
   if(g_daily_r <= InpDailyLossLimitR)      { Reject("daily_loss_limit"); return; }
   if(spread > InpMaxSpreadVsSl * stop_distance) { Reject("spread_vs_sl"); return; }
   if(spread > InpMaxSpreadAtr * atr)            { Reject("spread_vs_atr"); return; }

   long stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   if(stop_distance < stops_level * point)
     { Reject(StringFormat("stop_inside_broker_minimum(%d pts)", (int)stops_level)); return; }

   double risk_budget = g_initial_balance * InpRiskPercent / 100.0;
   double per_lot = MoneyPerLot(stop_distance);
   if(per_lot <= 0.0) { Reject("cannot_value_stop"); return; }

   double lots = NormaliseVolume(risk_budget / per_lot);
   if(lots <= 0.0) { Reject("min_lot_exceeds_risk_cap"); return; }

   double risk_money = lots * per_lot;
   if(risk_money > risk_budget + 0.01) { Reject("normalised_risk_over_cap"); return; }

   double margin;
   ENUM_ORDER_TYPE otype = (g_setup.direction == DIR_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   if(OrderCalcMargin(otype, _Symbol, lots, entry, margin) &&
      margin > AccountInfoDouble(ACCOUNT_MARGIN_FREE))
     { Reject("insufficient_margin"); return; }

   string summary = StringFormat("%s %.2f lots @ %.*f SL %.*f TP %.*f (risk %.2f, stop %.*f)",
                     g_setup.direction == DIR_BUY ? "BUY" : "SELL", lots,
                     digits, entry, digits, stop, digits, target, risk_money, digits, stop_distance);

   if(InpDryRun)
     {
      Print("DRY RUN would place: ", summary);
      ZeroSetup();
      return;
     }

   bool ok = (g_setup.direction == DIR_BUY)
             ? trade.Buy(lots, _Symbol, 0.0, NormalizeDouble(stop, digits), NormalizeDouble(target, digits),
                         StringFormat("v1.1 c%d", g_setup.confluence))
             : trade.Sell(lots, _Symbol, 0.0, NormalizeDouble(stop, digits), NormalizeDouble(target, digits),
                          StringFormat("v1.1 c%d", g_setup.confluence));

   if(!ok)
     {
      PrintFormat("ORDER FAILED (retcode %d %s): %s",
                  trade.ResultRetcode(), trade.ResultRetcodeDescription(), summary);
      ZeroSetup();
      return;
     }

   g_open_risk_money = risk_money;
   g_trades_today++;
   SaveState();
   Print("ORDER PLACED #", trade.ResultOrder(), ": ", summary);
   ZeroSetup();
  }

void Reject(const string reason)
  {
   Log("entry REJECTED (" + reason + ")");
   ZeroSetup();
  }

//+------------------------------------------------------------------+
//| Position tracking                                                 |
//+------------------------------------------------------------------+
int OpenPositions()
  {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == (long)InpMagic)
         count++;
     }
   return(count);
  }

void ReconcilePosition()
  {
   int n = OpenPositions();
   if(n > 0)
      PrintFormat("reconciled: %d existing position(s) adopted by magic %I64u", n, InpMagic);
  }

//+------------------------------------------------------------------+
//| Section 5.5 -- realised outcome bookkeeping                       |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagic) return;
   if(HistoryDealGetInteger(trans.deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;

   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                 + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
                 + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   double risk = (g_open_risk_money > 0.0) ? g_open_risk_money : 1.0;
   double r = profit / risk;

   g_daily_r += r;
   if(r < 0) g_losses_today++;
   g_cooldown_until = TimeCurrent() + InpCooldownBars * PeriodSeconds(PERIOD_M5);

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance >= g_initial_balance * InpTargetMultiplier)
     {
      g_target_reached = true;
      Print("ALERT target reached: closed balance ", balance, " -- entries blocked until reset");
     }
   g_open_risk_money = 0.0;
   SaveState();
   PrintFormat("position closed: %+.2f (%+.2fR), balance %.2f", profit, r, balance);
  }

//+------------------------------------------------------------------+
void RollBrokerDay()
  {
   MqlDateTime st;
   TimeToStruct(TimeCurrent(), st);
   st.hour = 0; st.min = 0; st.sec = 0;
   datetime today = StructToTime(st);
   if(today != g_broker_day)
     {
      g_broker_day = today;
      g_trades_today = 0;
      g_losses_today = 0;
      g_daily_r = 0.0;
      SaveState();
     }
  }

void UpdateEquityPeak()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_equity_peak) g_equity_peak = equity;
   if(g_equity_peak > 0.0 && !g_drawdown_locked)
     {
      double dd = 1.0 - equity / g_equity_peak;
      if(dd >= InpMaxPeakDrawdown)
        {
         g_drawdown_locked = true;
         PrintFormat("ALERT drawdown lock: equity %.2f is %.1f%% below peak %.2f -- "
                     "new entries disabled pending manual review",
                     equity, dd * 100.0, g_equity_peak);
         SaveState();
        }
     }
  }

//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime t = iTime(_Symbol, PERIOD_M5, 0);
   if(t == g_last_bar) return(false);
   g_last_bar = t;
   return(true);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   if(!IsNewBar()) return;
   if(Bars(_Symbol, PERIOD_M5) < InpEmaPeriod + 10) return;

   RollBrokerDay();
   UpdateEquityPeak();

   int regime = ComputeRegime();

   if(g_target_reached || g_drawdown_locked)
     {
      if(g_setup.active) ZeroSetup();
      SaveState();
      return;
     }

   DetectFvg();
   DetectSr();
   DetectOb();

   UpdateSetup(regime);
   ProcessZones(regime);

   SaveState();
  }
//+------------------------------------------------------------------+
