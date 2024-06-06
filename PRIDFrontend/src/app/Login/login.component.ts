import {Component} from "@angular/core"
import { NavigationExtras, Router } from "@angular/router"
import {LoginSerice} from "../service/login.service"
@Component({
    selector : "login",
    templateUrl : "login.component.html",
    styleUrls : ["./login.component.css"]
})

export class LoginComponent{
    email_address:String  = ""
    password:String  = ""
    login_validation_error : String = ""
    public constructor(private _login_service : LoginSerice,
        private _router : Router){

    }
    public login()
    {
        if(this.email_address != "" && this.password != "")
        {
            this._login_service.login(this.email_address,this.password).subscribe((response:any)=>{
                // console.log(response)
                if(response["login_sucess"])
                {
                    let navigationExtras: NavigationExtras = {
                        queryParams: {
                            username: this.email_address,
                        },skipLocationChange:true
                    }
                    this._router.navigate(["/videoprocess"],navigationExtras);
                    
                    }    else
                    this.login_validation_error = "Enter correct credential";
            });
        }else{
            this.login_validation_error = "Please Enter username and password"
        }
        
    }


    public redirect_to_demo() {
        
        let navigationExtras: NavigationExtras = {
            queryParams: {
                username: "prid",
            },skipLocationChange:true
        }
        this._router.navigate(["/person-reid-demo"],navigationExtras);
    }
}