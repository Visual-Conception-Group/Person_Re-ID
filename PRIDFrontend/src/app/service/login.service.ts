import {Injectable} from "@angular/core";
import {HttpClient, HttpHeaders} from "@angular/common/http"
@Injectable()
export class LoginSerice{
    httpurl : String = "http://reid.iiitd.edu.in:80"
    public constructor(private _httpclient : HttpClient){
    }
    httpHeaders = new HttpHeaders({
        'Access-Control-Allow-Origin': this.httpurl.toString()
      });
    
    options = {
        headers : this.httpHeaders
    }
    public login(username:String,password:String){
        return this._httpclient.post(this.httpurl+"/prid/api/v1/login",{
            "username" : username,
            "password" : password
        },
        this.options);
    }
}