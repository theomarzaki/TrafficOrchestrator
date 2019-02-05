const net = require('net');


const PORT = 8124;

class ToTCPServer {

    constructor() {
        this.server = net.createServer((socket) => {
            socket.on('data', (data) => {
                this.handleData(data.toString('utf8'), socket);
            });

            socket.on('error', (err) => {
                console.log("socket error ", err.toString());
                // when error occur we need to stop trying to send data to this socket
                const subscriptionId = socket.subscriptionId;
                toMessageHandler.clearTimers(subscriptionId);
                socket.destroy();
            });

            socket.on('end', () => {
                console.log("client disconnected");
            });
        });

        this.server.on('listening', () => {
            console.log(`TO TCP Server listening on port ${PORT}`);
        });
        this.server.on('connection', () => {
            console.log('Client connected');
        });
        this.server.on('close', () => {
            console.log('TO TCP Server closed');
        });
        this.server.on('error', (err) => {
            console.log('TO TCP server throw error', err);
        });
    }

    start() {
        this.server.listen(PORT, '0.0.0.0');
    }

    handleData(data, socket) {
        try {
            const message = JSON.parse(data);
            console.log("message Type : " +message.type);
            switch (message.type) {
                case "subscription_request":
                	console.log("in case");
                    this.handleSubscriptionRequest(message, socket);
                    break;
                default:
                    console.log(`received unsupported message type ${message}`);
                    socket.write("message.type should be subscription_request, unsubscription_request or maneuver_recommendation");
            }
        } catch (exception) {
            console.log(`received invalid json data ${data}`);
            socket.write(`Error invalid json ${exception.toString()}`);
        }

    }

    handleSubscriptionRequest(message, socket) {
    	console.log("received subscription request");
    	socket.write(`{"type":"subscription_response","context":"subscriptions","origin":"injector","version":"0.1.0","timestamp":1549366070634,"source_uuid":"injector","message":{"result":"accept","request_id":14289383,"subscription_id":1},"signature":"TEMPLATE"}\n`);
    	        let notification = '{"type":"notify_add","context":"subscriptions","origin":"gdm","version":"1.1.0","timestamp":1521632301738,"source_uuid":"gdm","message":{"subscription_id":12345,"ru_description_list":[{"type":"ru_description","context":"lane_merge","origin":"self","version":"0.2.1","timestamp":1521631164290,"source_uuid":"CAM","message":{"uuid":"OBU1","its_station_type":"passengerCar","connected":false,"position":{"latitude":421029701,"longitude":-86138355},"position_type":"raw_gnss","heading":0.00,"speed":1.0,"acceleration":0.0,"yaw_rate":-2.0,"size":{"length":40,"width":18,"height":15},"color":"0x000000","lane_position":0,"existence_probability":100,"confidence":{"position_semi_major_confidence":4095,"position_semi_minor_confidence":4095,"position_semi_major_orientation":3061,"heading":0,"speed":1,"acceleration":1,"yaw_rate":1,"size":{"length":1,"width":1,"height":1}}},"signature":"17"},{"type":"ru_description","context":"lane_merge","origin":"self","version":"0.2.1","timestamp":1521631164238,"source_uuid":"OBU9","message":{"uuid":"OBU9","its_station_type":"passengerCar","connected":true,"position":{"latitude":421029815.00,"longitude":-86137560.00},"position_type":"raw_gnss","heading":148.10,"speed":9.8,"acceleration":3.0,"yaw_rate":-147.0,"size":{"length":40,"width":18,"height":15},"color":"0x000000","lane_position":0,"existence_probability":100,"confidence":{"position_semi_major_confidence":4095,"position_semi_minor_confidence":4095,"position_semi_major_orientation":3061,"heading":0,"speed":1,"acceleration":1,"yaw_rate":1,"size":{"length":1,"width":1,"height":1}}},"signature":"4141383530"}]},"signature":"42"}';
        while (notification !== "") {
            const part = notification.substr(0, 1);
            notification = notification.substr(1);
            socket.write(part);
        }
        socket.write("\n");
        console.log("wrote notification in chunks");
    }

}

const tcp = new ToTCPServer();

tcp.start();