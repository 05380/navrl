// Auto-generated. Do not edit!

// (in-package navigation_runner.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class GetPolicyInferenceRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.state = null;
      this.state_shape = null;
      this.lidar = null;
      this.lidar_shape = null;
      this.direction = null;
      this.direction_shape = null;
      this.dynamic_obstacle = null;
      this.dynamic_obstacle_shape = null;
    }
    else {
      if (initObj.hasOwnProperty('state')) {
        this.state = initObj.state
      }
      else {
        this.state = [];
      }
      if (initObj.hasOwnProperty('state_shape')) {
        this.state_shape = initObj.state_shape
      }
      else {
        this.state_shape = [];
      }
      if (initObj.hasOwnProperty('lidar')) {
        this.lidar = initObj.lidar
      }
      else {
        this.lidar = [];
      }
      if (initObj.hasOwnProperty('lidar_shape')) {
        this.lidar_shape = initObj.lidar_shape
      }
      else {
        this.lidar_shape = [];
      }
      if (initObj.hasOwnProperty('direction')) {
        this.direction = initObj.direction
      }
      else {
        this.direction = [];
      }
      if (initObj.hasOwnProperty('direction_shape')) {
        this.direction_shape = initObj.direction_shape
      }
      else {
        this.direction_shape = [];
      }
      if (initObj.hasOwnProperty('dynamic_obstacle')) {
        this.dynamic_obstacle = initObj.dynamic_obstacle
      }
      else {
        this.dynamic_obstacle = [];
      }
      if (initObj.hasOwnProperty('dynamic_obstacle_shape')) {
        this.dynamic_obstacle_shape = initObj.dynamic_obstacle_shape
      }
      else {
        this.dynamic_obstacle_shape = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetPolicyInferenceRequest
    // Serialize message field [state]
    bufferOffset = _arraySerializer.float64(obj.state, buffer, bufferOffset, null);
    // Serialize message field [state_shape]
    bufferOffset = _arraySerializer.int32(obj.state_shape, buffer, bufferOffset, null);
    // Serialize message field [lidar]
    bufferOffset = _arraySerializer.float64(obj.lidar, buffer, bufferOffset, null);
    // Serialize message field [lidar_shape]
    bufferOffset = _arraySerializer.int32(obj.lidar_shape, buffer, bufferOffset, null);
    // Serialize message field [direction]
    bufferOffset = _arraySerializer.float64(obj.direction, buffer, bufferOffset, null);
    // Serialize message field [direction_shape]
    bufferOffset = _arraySerializer.int32(obj.direction_shape, buffer, bufferOffset, null);
    // Serialize message field [dynamic_obstacle]
    bufferOffset = _arraySerializer.float64(obj.dynamic_obstacle, buffer, bufferOffset, null);
    // Serialize message field [dynamic_obstacle_shape]
    bufferOffset = _arraySerializer.int32(obj.dynamic_obstacle_shape, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetPolicyInferenceRequest
    let len;
    let data = new GetPolicyInferenceRequest(null);
    // Deserialize message field [state]
    data.state = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [state_shape]
    data.state_shape = _arrayDeserializer.int32(buffer, bufferOffset, null)
    // Deserialize message field [lidar]
    data.lidar = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [lidar_shape]
    data.lidar_shape = _arrayDeserializer.int32(buffer, bufferOffset, null)
    // Deserialize message field [direction]
    data.direction = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [direction_shape]
    data.direction_shape = _arrayDeserializer.int32(buffer, bufferOffset, null)
    // Deserialize message field [dynamic_obstacle]
    data.dynamic_obstacle = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [dynamic_obstacle_shape]
    data.dynamic_obstacle_shape = _arrayDeserializer.int32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 8 * object.state.length;
    length += 4 * object.state_shape.length;
    length += 8 * object.lidar.length;
    length += 4 * object.lidar_shape.length;
    length += 8 * object.direction.length;
    length += 4 * object.direction_shape.length;
    length += 8 * object.dynamic_obstacle.length;
    length += 4 * object.dynamic_obstacle_shape.length;
    return length + 32;
  }

  static datatype() {
    // Returns string type for a service object
    return 'navigation_runner/GetPolicyInferenceRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'eba1c092147ccc132b4778420f8a4a24';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64[] state
    int32[] state_shape
    float64[] lidar
    int32[] lidar_shape
    float64[] direction
    int32[] direction_shape
    float64[] dynamic_obstacle
    int32[] dynamic_obstacle_shape
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetPolicyInferenceRequest(null);
    if (msg.state !== undefined) {
      resolved.state = msg.state;
    }
    else {
      resolved.state = []
    }

    if (msg.state_shape !== undefined) {
      resolved.state_shape = msg.state_shape;
    }
    else {
      resolved.state_shape = []
    }

    if (msg.lidar !== undefined) {
      resolved.lidar = msg.lidar;
    }
    else {
      resolved.lidar = []
    }

    if (msg.lidar_shape !== undefined) {
      resolved.lidar_shape = msg.lidar_shape;
    }
    else {
      resolved.lidar_shape = []
    }

    if (msg.direction !== undefined) {
      resolved.direction = msg.direction;
    }
    else {
      resolved.direction = []
    }

    if (msg.direction_shape !== undefined) {
      resolved.direction_shape = msg.direction_shape;
    }
    else {
      resolved.direction_shape = []
    }

    if (msg.dynamic_obstacle !== undefined) {
      resolved.dynamic_obstacle = msg.dynamic_obstacle;
    }
    else {
      resolved.dynamic_obstacle = []
    }

    if (msg.dynamic_obstacle_shape !== undefined) {
      resolved.dynamic_obstacle_shape = msg.dynamic_obstacle_shape;
    }
    else {
      resolved.dynamic_obstacle_shape = []
    }

    return resolved;
    }
};

class GetPolicyInferenceResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.action = null;
    }
    else {
      if (initObj.hasOwnProperty('action')) {
        this.action = initObj.action
      }
      else {
        this.action = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetPolicyInferenceResponse
    // Serialize message field [action]
    bufferOffset = _arraySerializer.float64(obj.action, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetPolicyInferenceResponse
    let len;
    let data = new GetPolicyInferenceResponse(null);
    // Deserialize message field [action]
    data.action = _arrayDeserializer.float64(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 8 * object.action.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'navigation_runner/GetPolicyInferenceResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '79f44d272f2ebe04451185b0dea57684';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float64[] action
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetPolicyInferenceResponse(null);
    if (msg.action !== undefined) {
      resolved.action = msg.action;
    }
    else {
      resolved.action = []
    }

    return resolved;
    }
};

module.exports = {
  Request: GetPolicyInferenceRequest,
  Response: GetPolicyInferenceResponse,
  md5sum() { return 'a4b6b43418ed234e9de75a5ccb110333'; },
  datatype() { return 'navigation_runner/GetPolicyInference'; }
};
