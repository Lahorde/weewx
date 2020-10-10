#
#    Copyright (c) 2009-2020 Tom Keffer <tkeffer@gmail.com>
#
#    See the file LICENSE.txt for your full rights.
#
"""A set of XTypes extensions for derived weather-related types."""
from __future__ import absolute_import

import logging

import weedb
import weeutil.logger
import weewx.engine
import weewx.units
import weewx.wxformulas
import weewx.xtypes
from weeutil.weeutil import to_int, to_float, to_bool
from weewx.units import ValueTuple, mps_to_mph, kph_to_mph, METER_PER_FOOT, CtoF

log = logging.getLogger(__name__)

first_time = True


#
# ######################## Class STDWXXTypes ##############################
#

class StdWXXTypes(weewx.xtypes.XType, weewx.engine.StdService):
    """Weather extensions to the WeeWX xtype system that are relatively simple. These types
    are generally stateless, such as dewpoint, heatindex, etc. """

    def __init__(self, engine, config_dict):
        """Initialize an instance of StdWXXTypes"""
        super(StdWXXTypes, self).__init__(engine, config_dict)

        self.altitude_vt = engine.stn_info.altitude_vt
        self.latitude_f = engine.stn_info.latitude_f
        self.longitude_f = engine.stn_info.longitude_f
        try:
            calc_dict = config_dict['StdWXCalculate']
        except KeyError:
            calc_dict = {}
            # window of time for evapotranspiration calculation, in seconds
        self.et_period = to_int(calc_dict.get('et_period', 3600))
        # atmospheric transmission coefficient [0.7-0.91]
        self.atc = to_float(calc_dict.get('atc', 0.8))
        # Fail hard if out of range:
        if not 0.7 <= self.atc <= 0.91:
            raise weewx.ViolatedPrecondition("Atmospheric transmission coefficient (%f) "
                                             "out of range [.7-.91]" % self.atc)
        # atmospheric turbidity (2=clear, 4-5=smoggy)
        self.nfac = to_float(calc_dict.get('nfac', 2))
        # height above ground at which wind is measured, in meters
        self.wind_height = to_float(calc_dict.get('wind_height', 2.0))
        # Adjust wind direction to null, if the wind speed is zero:
        self.ignore_zero_wind = to_bool(calc_dict.get('ignore_zero_wind', False))

        self.maxSolarRad_algo = calc_dict.get('Algorithms',
                                              {'maxSolarRad': 'rs'}).get('maxSolarRad',
                                                                         'rs').lower()

        # Add myself to the XTypes system
        weewx.xtypes.xtypes.append(self)

    def shut_down(self):
        """Engine shutting down. """
        # Remove myself from the XTypes system:
        weewx.xtypes.xtypes.remove(self)

    def get_scalar(self, obs_type, record, db_manager):
        """Invoke the proper method for the desired observation type."""
        try:
            # Form the method name and call it with arguments
            return getattr(self, 'calc_%s' % obs_type)(obs_type, record, db_manager)
        except AttributeError:
            raise weewx.UnknownType(obs_type)

    def calc_windDir(self, key, data, db_manager):
        # Return the current wind direction if windSpeed is non-zero, otherwise, None
        if 'windSpeed' not in data:
            raise weewx.CannotCalculate
        if self.ignore_zero_wind or data['windSpeed']:
            val = data.get('windDir')
        else:
            val = None
        return ValueTuple(val, 'degree_compass', 'group_direction')

    def calc_windGustDir(self, key, data, db_manager):
        # Return the current gust direction if windGust is non-zero, otherwise, None
        if 'windGust' not in data:
            raise weewx.CannotCalculate
        if self.ignore_zero_wind or data['windGust']:
            val = data.get('windGustDir')
        else:
            val = None
        return ValueTuple(val, 'degree_compass', 'group_direction')

    def calc_maxSolarRad(self, key, data, db_manager):
        altitude_m = weewx.units.convert(self.altitude_vt, 'meter')[0]
        if self.maxSolarRad_algo == 'bras':
            val = weewx.wxformulas.solar_rad_Bras(self.latitude_f, self.longitude_f, altitude_m,
                                                  data['dateTime'], self.nfac)
        elif self.maxSolarRad_algo == 'rs':
            val = weewx.wxformulas.solar_rad_RS(self.latitude_f, self.longitude_f, altitude_m,
                                                data['dateTime'], self.atc)
        else:
            raise weewx.ViolatedPrecondition("Unknown solar algorithm '%s'"
                                             % self.maxSolarRad_algo)
        return ValueTuple(val, 'watt_per_meter_squared', 'group_radiation')

    def calc_cloudbase(self, key, data, db_manager):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        # Convert altitude to the same unit system as the incoming record
        altitude = weewx.units.convertStd(self.altitude_vt, data['usUnits'])
        # Use the appropriate formula
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.cloudbase_US(data['outTemp'],
                                                data['outHumidity'], altitude[0])
            u = 'foot'
        else:
            val = weewx.wxformulas.cloudbase_Metric(data['outTemp'],
                                                    data['outHumidity'], altitude[0])
            u = 'meter'
        return ValueTuple(val, u, 'group_altitude')

    def calc_ET(self, key, data, db_manager):
        """Get maximum and minimum temperatures and average radiation and wind speed for the
        indicated period then calculate the amount of evapotranspiration during the interval.
        Convert to US units if necessary since this service operates in US unit system.
        """

        if 'interval' not in data:
            # This will cause LOOP data not to be processed.
            raise weewx.CannotCalculate(key)

        interval = data['interval']
        end_ts = data['dateTime']
        start_ts = end_ts - self.et_period
        try:
            r = db_manager.getSql("SELECT MAX(outTemp), MIN(outTemp), "
                                  "AVG(radiation), AVG(windSpeed), "
                                  "MAX(outHumidity), MIN(outHumidity), "
                                  "MAX(usUnits), MIN(usUnits) FROM %s "
                                  "WHERE dateTime>? AND dateTime <=?"
                                  % db_manager.table_name, (start_ts, end_ts))
        except weedb.DatabaseError:
            return ValueTuple(None, None, None)

        # Make sure everything is there:
        if r is None or None in r:
            return ValueTuple(None, None, None)

        # Unpack the results
        T_max, T_min, rad_avg, wind_avg, rh_max, rh_min, std_unit_min, std_unit_max = r

        # Check for mixed units
        if std_unit_min != std_unit_max:
            log.info("Mixed unit system not allowed in ET calculation. Skipped.")
            return ValueTuple(None, None, None)
        std_unit = std_unit_min
        if std_unit == weewx.METRIC or std_unit == weewx.METRICWX:
            T_max = CtoF(T_max)
            T_min = CtoF(T_min)
            if std_unit == weewx.METRICWX:
                wind_avg = mps_to_mph(wind_avg)
            else:
                wind_avg = kph_to_mph(wind_avg)
        # Wind height is in meters, so convert it:
        height_ft = self.wind_height / METER_PER_FOOT
        # Get altitude in feet
        altitude_ft = weewx.units.convert(self.altitude_vt, 'foot')[0]

        try:
            ET_rate = weewx.wxformulas.evapotranspiration_US(
                T_min, T_max, rh_min, rh_max, rad_avg, wind_avg, height_ft,
                self.latitude_f, self.longitude_f, altitude_ft, end_ts)
        except ValueError as e:
            log.error("Calculation of evapotranspiration failed: %s", e)
            weeutil.logger.log_traceback(log.error)
            ET_inch = None
        else:
            # The formula returns inches/hour. We need the total ET over the interval, so multiply
            # by the length of the interval in hours. Remember that 'interval' is actually in
            # minutes.
            ET_inch = ET_rate * interval / 60.0 if ET_rate is not None else None

        # Convert back to the unit system of the incoming record:
        ET = weewx.units.convertStd((ET_inch, 'inch', 'group_rain'), data['usUnits'])
        return ET

    @staticmethod
    def calc_dewpoint(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.dewpointF(data['outTemp'], data['outHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.dewpointC(data['outTemp'], data['outHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_inDewpoint(key, data, db_manager=None):
        if 'inTemp' not in data or 'inHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.dewpointF(data['inTemp'], data['inHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.dewpointC(data['inTemp'], data['inHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_windchill(key, data, db_manager=None):
        if 'outTemp' not in data or 'windSpeed' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.windchillF(data['outTemp'], data['windSpeed'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.windchillC(data['outTemp'], data['windSpeed'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_heatindex(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.heatindexF(data['outTemp'], data['outHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.heatindexC(data['outTemp'], data['outHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_humidex(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.humidexF(data['outTemp'], data['outHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.humidexC(data['outTemp'], data['outHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_appTemp(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data or 'windSpeed' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.apptempF(data['outTemp'], data['outHumidity'],
                                            data['windSpeed'])
            u = 'degree_F'
        else:
            # The metric equivalent needs wind speed in mps. Convert.
            windspeed_vt = weewx.units.as_value_tuple(data, 'windSpeed')
            windspeed_mps = weewx.units.convert(windspeed_vt, 'meter_per_second')[0]
            val = weewx.wxformulas.apptempC(data['outTemp'], data['outHumidity'], windspeed_mps)
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_beaufort(key, data, db_manager=None):
        global first_time
        if first_time:
            print("Type beaufort has been deprecated. Use unit beaufort instead.")
            log.info("Type beaufort has been deprecated. Use unit beaufort instead.")
            first_time = False
        if 'windSpeed' not in data:
            raise weewx.CannotCalculate
        windspeed_vt = weewx.units.as_value_tuple(data, 'windSpeed')
        windspeed_kn = weewx.units.convert(windspeed_vt, 'knot')[0]
        return ValueTuple(weewx.wxformulas.beaufort(windspeed_kn), None, None)

    @staticmethod
    def calc_windrun(key, data, db_manager=None):
        """Calculate wind run. Requires key 'interval'"""
        if 'windSpeed' not in data or 'interval' not in data:
            raise weewx.CannotCalculate(key)

        if data['windSpeed'] is not None:
            if data['usUnits'] == weewx.US:
                val = data['windSpeed'] * data['interval'] / 60.0
                u = 'mile'
            elif data['usUnits'] == weewx.METRIC:
                val = data['windSpeed'] * data['interval'] / 60.0
                u = 'km'
            elif data['usUnits'] == weewx.METRICWX:
                val = data['windSpeed'] * data['interval'] * 60.0 / 1000.0
                u = 'km'
            else:
                raise weewx.ViolatedPrecondition("Unknown unit system %s" % data['usUnits'])
        else:
            val = None
            u = 'mile'
        return weewx.units.convertStd((val, u, 'group_distance'), data['usUnits'])


#
# ######################## Class PressureCooker ##############################
#

class PressureCooker(weewx.xtypes.XType, weewx.engine.StdService):
    """Pressure related extensions to the WeeWX type system. """

    def __init__(self, engine, config_dict):
        """Initialize the PressureCooker. """
        super(PressureCooker, self).__init__(engine, config_dict)

        self.altitude_vt = engine.stn_info.altitude_vt
        try:
            calc_dict = config_dict['StdWXCalculate']
        except KeyError:
            calc_dict = {}

        self.max_delta_12h = to_float(calc_dict.get('max_delta_12h', 1800))
        self.altimeter_algorithm = calc_dict.get('Algorithms',
                                                 {'altimeter': 'aaASOS'}).get('altimeter',
                                                                              'aaASOS')

        # Algorithms can be abbreviated without the prefix 'aa':
        if not self.altimeter_algorithm.startswith('aa'):
            self.altimeter_algorithm = 'aa%s' % self.altimeter_algorithm

        # Timestamp (roughly) 12 hours ago
        self.ts_12h = None
        # Temperature 12 hours ago as a ValueTuple
        self.temp_12h_vt = None

        # Add myself to the XTypes system
        weewx.xtypes.xtypes.append(self)

    def shut_down(self):
        """Engine shutting down. """
        # Remove myself from the XTypes system:
        weewx.xtypes.xtypes.remove(self)

    def _get_temperature_12h(self, ts, dbmanager):
        """Get the temperature as a ValueTuple from 12 hours ago.  The value will
         be None if no temperature is available.
         """

        ts_12h = ts - 12 * 3600

        # Look up the temperature 12h ago if this is the first time through,
        # or we don't have a usable temperature, or the old temperature is too stale.
        if self.ts_12h is None \
                or self.temp_12h_vt is None \
                or abs(self.ts_12h - ts_12h) < self.max_delta_12h:
            # Hit the database to get a newer temperature.
            record = dbmanager.getRecord(ts_12h, max_delta=self.max_delta_12h)
            if record and 'outTemp' in record:
                # Figure out what unit the record is in ...
                unit = weewx.units.getStandardUnitType(record['usUnits'], 'outTemp')
                # ... then form a ValueTuple.
                self.temp_12h_vt = weewx.units.ValueTuple(record['outTemp'], *unit)
            else:
                # Invalidate the temperature ValueTuple from 12h ago
                self.temp_12h_vt = None
            # Save the timestamp
            self.ts_12h = ts_12h

        return self.temp_12h_vt

    def get_scalar(self, key, record, dbmanager):
        if key == 'pressure':
            return self.pressure(record, dbmanager)
        elif key == 'altimeter':
            return self.altimeter(record)
        elif key == 'barometer':
            return self.barometer(record)
        else:
            raise weewx.UnknownType(key)

    def pressure(self, record, dbmanager):
        """Calculate the observation type 'pressure'."""

        # All of the following keys are required:
        if any(key not in record for key in ['usUnits', 'outTemp', 'barometer', 'outHumidity']):
            raise weewx.CannotCalculate('pressure')

        # Get the temperature in Fahrenheit from 12 hours ago
        temp_12h_vt = self._get_temperature_12h(record['dateTime'], dbmanager)
        if temp_12h_vt is None \
                or temp_12h_vt[0] is None \
                or record['outTemp'] is None \
                or record['barometer'] is None \
                or record['outHumidity'] is None:
            pressure = None
        else:
            # The following requires everything to be in US Customary units.
            # Rather than convert the whole record, just convert what we need:
            record_US = weewx.units.to_US({'usUnits': record['usUnits'],
                                           'outTemp': record['outTemp'],
                                           'barometer': record['barometer'],
                                           'outHumidity': record['outHumidity']})
            # Get the altitude in feet
            altitude_ft = weewx.units.convert(self.altitude_vt, "foot")
            # The outside temperature in F.
            temp_12h_F = weewx.units.convert(temp_12h_vt, "degree_F")
            pressure = weewx.uwxutils.uWxUtilsVP.SeaLevelToSensorPressure_12(
                record_US['barometer'],
                altitude_ft[0],
                record_US['outTemp'],
                temp_12h_F[0],
                record_US['outHumidity']
            )

        # Convert to target unit system and return
        return weewx.units.convertStd((pressure, 'inHg', 'group_pressure'), record['usUnits'])

    def altimeter(self, record):
        """Calculate the observation type 'altimeter'."""
        if 'pressure' not in record:
            raise weewx.CannotCalculate('altimeter')

        # Convert altitude to same unit system of the incoming record
        altitude = weewx.units.convertStd(self.altitude_vt, record['usUnits'])

        # Figure out which altimeter formula to use, and what unit the results will be in:
        if record['usUnits'] == weewx.US:
            formula = weewx.wxformulas.altimeter_pressure_US
            u = 'inHg'
        else:
            formula = weewx.wxformulas.altimeter_pressure_Metric
            u = 'mbar'
        # Apply the formula
        altimeter = formula(record['pressure'], altitude[0], self.altimeter_algorithm)
        # Convert to the target unit system
        return weewx.units.convertStd((altimeter, u, 'group_pressure'), record['usUnits'])

    def barometer(self, record):
        """Calculate the observation type 'barometer'"""

        if 'pressure' not in record or 'outTemp' not in record:
            raise weewx.CannotCalculate('barometer')

        # Convert altitude to same unit system of the incoming record
        altitude = weewx.units.convertStd(self.altitude_vt, record['usUnits'])

        # Figure out what barometer formula to use:
        if record['usUnits'] == weewx.US:
            formula = weewx.wxformulas.sealevel_pressure_US
            u = 'inHg'
        else:
            formula = weewx.wxformulas.sealevel_pressure_Metric
            u = 'mbar'
        # Apply the formula
        barometer = formula(record['pressure'], altitude[0], record['outTemp'])
        # Convert to the target unit system:
        return weewx.units.convertStd((barometer, u, 'group_pressure'), record['usUnits'])


#
# ######################## Class RainRater ##############################
#

class RainRater(weewx.xtypes.XType, weewx.engine.StdService):
    """"An extension to the WeeWX type system for calculating rainRate"""

    # def __init__(self, rain_period, retain_period):
    def __init__(self, engine, config_dict):
        """Initialize the RainRater."""
        super(RainRater, self).__init__(engine, config_dict)

        try:
            calc_dict = config_dict['StdWXCalculate']
        except KeyError:
            calc_dict = {}
        self.rain_period = to_int(calc_dict.get('rain_period', 900))
        self.retain_period = to_int(calc_dict.get('retain_period', 930))

        self.rain_events = None
        self.unit_system = None

        self.bind(weewx.NEW_LOOP_PACKET, self.new_loop_packet)

        # Add myself to the XTypes system
        weewx.xtypes.xtypes.append(self)

    def shut_down(self):
        """Engine shutting down. """
        # Remove myself from the XTypes system:
        weewx.xtypes.xtypes.remove(self)

    def new_loop_packet(self, event):
        self.add_loop_packet(event.packet)
        
    def add_loop_packet(self, packet):
        """Process LOOP packets, adding them to the list of recent rain events."""
        # Was there any rain? If so, convert the rain to the unit system we are using,
        # then intern it
        if 'rain' in packet and packet['rain']:
            if self.unit_system is None:
                # Adopt the unit system of the first record.
                self.unit_system = packet['usUnits']
            if self.rain_events is None:
                self._setup(packet['dateTime'])
            # Get the unit system and group of the incoming rain. In theory, this should be
            # the same as self.unit_system, but ...
            u, g = weewx.units.getStandardUnitType(packet['usUnits'], 'rain')
            # Convert to the unit system that we are using
            rain = weewx.units.convertStd((packet['rain'], u, g), self.unit_system)[0]
            # Add it to the list of rain events
            self.rain_events.append((packet['dateTime'], rain))

        if self.rain_events:
            # Trim any old packets:
            self.rain_events = [x for x in self.rain_events
                                if x[0] >= packet['dateTime'] - self.rain_period]

    def get_scalar(self, key, record, db_manager):
        """Calculate the rainRate"""
        if key != 'rainRate':
            raise weewx.UnknownType(key)

        if self.rain_events is None:
            self._setup(record['dateTime'], db_manager)

        # Sum the rain events within the time window...
        rainsum = sum(x[1] for x in self.rain_events
                      if x[0] > record['dateTime'] - self.rain_period)
        # ...then divide by the period and scale to an hour
        val = 3600 * rainsum / self.rain_period
        # Get the unit and unit group for rainRate
        u, g = weewx.units.getStandardUnitType(self.unit_system, 'rainRate')
        # Form a ValueTuple, then convert it to the unit system of the incoming record
        rr = weewx.units.convertStd(ValueTuple(val, u, g), record['usUnits'])
        return rr

    def _setup(self, stop_ts, db_manager=None):
        """Initialize the rain event list"""
        if self.rain_events is None:
            self.rain_events = []
        if db_manager is None:
            db_manager = self.engine.db_binder.get_manager(data_binding='wx_binding',
                                                           initialize=True)

            
        start_ts = stop_ts - self.retain_period
        # Get all rain events since the window start from the database. Put it in
        # a 'try' block because the database may not have a 'rain' field.
        try:
            for row in db_manager.genSql("SELECT dateTime, usUnits, rain FROM %s "
                                         "WHERE dateTime>? AND dateTime<=?;"
                                         % db_manager.table_name, (start_ts, stop_ts)):
                # Unpack the row:
                time_ts, unit_system, rain = row
                self.add_loop_packet({'dateTime': time_ts, 'usUnits': unit_system, 'rain': rain})
        except weedb.DatabaseError as e:
            log.debug("Database error while initializing rainRate: '%s'" % e)
